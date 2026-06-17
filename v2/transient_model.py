"""transient_model.py — 7-DOF vehicle ride model (time-domain).

A step up from the quasi-steady-state ``LapSimulator``: instead of treating
every point as an already-settled equilibrium, this integrates the chassis'
vertical dynamics through time, so **dampers finally do work**.

Seven degrees of freedom (deviations from the static equilibrium):
    sprung mass   : heave z_s, pitch theta, roll phi          (3)
    unsprung mass : vertical hop of each wheel z_u[FL,FR,RL,RR] (4)

Excitation is the inertial load transfer from the lap: a longitudinal
acceleration ax(t) feeds a pitch moment, a lateral acceleration ay(t) feeds a
roll moment. Each corner carries a spring (wheel rate), an **asymmetric**
damper (bump vs rebound coefficient — selected on the sign of the shaft
velocity, which is the whole reason dampers matter), an anti-roll bar coupling
the two wheels of an axle, and a vertical tire spring to the road.

Coordinates are deviations from the static-loaded equilibrium, so gravity is
already balanced and never appears in the equations of motion; the dynamic
tire normal load is ``N_static + k_tire * (z_road - z_u)``.

This module is intentionally self-contained — it reads a ``VehicleConfig`` but
adds no new fields to it, so the rest of the pipeline (export, optimiser,
tests) is untouched while we develop the transient path.

Corner order is fixed everywhere: [FL, FR, RL, RR].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import VehicleConfig

G = 9.81
_ORDER = ("FL", "FR", "RL", "RR")


@dataclass
class RideModelParams:
    """Flattened, per-corner physical parameters the integrator needs.

    Built from a :class:`VehicleConfig` via :meth:`from_config`. Tire vertical
    stiffness and the sprung-mass inertias are not in ``VehicleConfig`` (the
    QSS path never needed them), so they are derived/estimated here and can be
    overridden by the caller for studies."""

    m_s: float                 # total sprung mass [kg]
    I_pitch: float             # sprung pitch inertia [kg m^2]
    I_roll: float              # sprung roll inertia  [kg m^2]
    h_cg: float                # CG height above ground [m] (inertial lever)
    x: np.ndarray              # signed longitudinal arm of each corner [m] (+front)
    y: np.ndarray              # signed lateral arm of each corner [m] (+left)
    m_u: np.ndarray            # unsprung mass per corner [kg]
    k_s: np.ndarray            # suspension wheel rate per corner [N/m]
    c_bump: np.ndarray         # damper bump (compression) coeff per corner [N s/m]
    c_reb: np.ndarray          # damper rebound (extension) coeff per corner [N s/m]
    k_arb_f: float             # front ARB equivalent wheel rate [N/m]
    k_arb_r: float             # rear ARB equivalent wheel rate [N/m]
    k_t: np.ndarray            # tire vertical stiffness per corner [N/m]
    n_static: np.ndarray       # static vertical load per corner [N]

    @classmethod
    def from_config(cls, cfg: VehicleConfig, k_tire: float = 200_000.0,
                    pitch_gyr: float | None = None,
                    roll_gyr: float | None = None) -> "RideModelParams":
        m_uf, m_ur = cfg.unsprung_mass_f, cfg.unsprung_mass_r   # per axle
        m_u = np.array([m_uf / 2, m_uf / 2, m_ur / 2, m_ur / 2])
        m_s = cfg.mass - (m_uf + m_ur)

        # CG longitudinal location: front axle carries weight_dist_f of weight,
        # so the CG sits (1-weight_dist_f)*L behind the front axle.
        l_f = (1.0 - cfg.weight_dist_f) * cfg.wheelbase
        l_r = cfg.weight_dist_f * cfg.wheelbase
        x = np.array([+l_f, +l_f, -l_r, -l_r])
        y = np.array([+cfg.track_f / 2, -cfg.track_f / 2,
                      +cfg.track_r / 2, -cfg.track_r / 2])

        # Inertia estimates (radius of gyration). Defaults give a realistic
        # pitch/roll natural frequency; override for a measured car.
        kp = pitch_gyr if pitch_gyr is not None else np.sqrt(l_f * l_r)
        t_avg = 0.5 * (cfg.track_f + cfg.track_r)
        kr = roll_gyr if roll_gyr is not None else 0.5 * t_avg
        I_pitch = m_s * kp * kp
        I_roll = m_s * kr * kr

        k_s = np.array([
            cfg.spring_k_f * cfg.motion_ratio_f ** 2,
            cfg.spring_k_f * cfg.motion_ratio_f ** 2,
            cfg.spring_k_r * cfg.motion_ratio_r ** 2,
            cfg.spring_k_r * cfg.motion_ratio_r ** 2,
        ])
        c_bump = np.array([cfg.damper_bump_f, cfg.damper_bump_f,
                           cfg.damper_bump_r, cfg.damper_bump_r])
        c_reb = np.array([cfg.damper_rebound_f, cfg.damper_rebound_f,
                          cfg.damper_rebound_r, cfg.damper_rebound_r])

        f_axle = cfg.mass * G * cfg.weight_dist_f
        r_axle = cfg.mass * G * (1.0 - cfg.weight_dist_f)
        n_static = np.array([f_axle / 2, f_axle / 2, r_axle / 2, r_axle / 2])

        return cls(
            m_s=m_s, I_pitch=I_pitch, I_roll=I_roll, h_cg=cfg.cg_height,
            x=x, y=y, m_u=m_u, k_s=k_s, c_bump=c_bump, c_reb=c_reb,
            k_arb_f=cfg.arb_k_f, k_arb_r=cfg.arb_k_r,
            k_t=np.full(4, float(k_tire)), n_static=n_static,
        )


@dataclass
class RideResult:
    """Output of :meth:`SevenDOFRideModel.simulate`."""

    t: np.ndarray              # time samples [s]
    z_s: np.ndarray            # sprung heave [m]
    pitch: np.ndarray          # pitch angle [rad] (+ = nose up)
    roll: np.ndarray           # roll angle [rad]  (+ = left side up)
    z_u: np.ndarray            # unsprung hop, shape (n, 4) [m]
    tire_load: np.ndarray      # dynamic tire normal load, shape (n, 4) [N]


class SevenDOFRideModel:
    """Time-domain 7-DOF ride/handling-load model.

    State vector (14): [z_s, pitch, roll, z_u(4), then the 7 velocities].
    """

    def __init__(self, params: RideModelParams, v_eps: float = 0.02):
        self.p = params
        # Velocity band over which bump<->rebound damping blends. A real damper
        # transitions smoothly through zero shaft speed; this regularisation
        # also removes the hard switch that would otherwise slow (or stall)
        # stiff/long integrations. Small relative to corner-transition speeds.
        self.v_eps = float(v_eps)

    # ------------------------------------------------------------------ #
    def _corner_kinematics(self, z_s, pitch, roll, vz_s, vpitch, vroll):
        """Sprung-mass vertical position/velocity at each contact patch."""
        p = self.p
        z_sc = z_s + p.x * pitch + p.y * roll
        v_sc = vz_s + p.x * vpitch + p.y * vroll
        return z_sc, v_sc

    def _suspension_forces(self, z_sc, v_sc, z_u, v_u):
        """Per-corner vertical suspension force on the body (+up), and the ARB
        contribution, given sprung-side and unsprung kinematics."""
        p = self.p
        # Extension-positive deflection / velocity of each strut.
        defl = z_sc - z_u
        vel = v_sc - v_u
        # Asymmetric damping: bump (compression, vel < 0) vs rebound
        # (extension, vel > 0), blended smoothly through zero shaft speed.
        w = 0.5 * (1.0 - np.tanh(vel / self.v_eps))   # ->1 in bump, ->0 rebound
        c = p.c_reb + (p.c_bump - p.c_reb) * w
        f_spring = -p.k_s * defl
        f_damp = -c * vel
        # ARB resists the across-axle deflection difference (left minus right).
        d_f = defl[0] - defl[1]
        d_r = defl[2] - defl[3]
        f_arb = np.array([
            -p.k_arb_f * d_f, +p.k_arb_f * d_f,
            -p.k_arb_r * d_r, +p.k_arb_r * d_r,
        ])
        return f_spring + f_damp + f_arb

    def derivatives(self, t, state, ax_fn, ay_fn, road_fn):
        p = self.p
        z_s, pitch, roll = state[0], state[1], state[2]
        z_u = state[3:7]
        vz_s, vpitch, vroll = state[7], state[8], state[9]
        v_u = state[10:14]

        z_sc, v_sc = self._corner_kinematics(z_s, pitch, roll,
                                             vz_s, vpitch, vroll)
        f_body = self._suspension_forces(z_sc, v_sc, z_u, v_u)

        ax, ay = float(ax_fn(t)), float(ay_fn(t))
        z_road = road_fn(t)
        # Inertial load-transfer moments (sprung mass, acting at CG height).
        m_pitch = p.m_s * ax * p.h_cg     # +ax -> nose up
        m_roll = p.m_s * ay * p.h_cg      # +ay (left) -> left side up

        acc_z = f_body.sum() / p.m_s
        acc_pitch = (np.dot(p.x, f_body) + m_pitch) / p.I_pitch
        acc_roll = (np.dot(p.y, f_body) + m_roll) / p.I_roll

        # Unsprung: suspension reacts opposite the body; the tire spring pushes
        # up but can only PUSH — once a wheel lifts (total normal load would go
        # negative) it leaves the ground and contributes no force.
        n_total = np.maximum(p.n_static + p.k_t * (z_road - z_u), 0.0)
        f_tire = n_total - p.n_static     # dynamic deviation, with liftoff
        acc_u = (-f_body + f_tire) / p.m_u

        return np.concatenate((
            [vz_s, vpitch, vroll], v_u,
            [acc_z, acc_pitch, acc_roll], acc_u,
        ))

    # ------------------------------------------------------------------ #
    def tire_loads(self, z_u, z_road=0.0):
        """Tire normal load [FL,FR,RL,RR] from unsprung deflection, clamped at
        zero (a lifted wheel carries no load)."""
        return np.maximum(
            self.p.n_static + self.p.k_t * (z_road - np.asarray(z_u)), 0.0)

    def simulate(self, t_end, ax, ay, road=None, dt=0.001, state0=None,
                 t_eval=None, method="RK45", max_step=None):
        """Integrate the ride model.

        ``ax``/``ay`` may be scalars or callables of time [m/s^2]; ``road`` is
        an optional callable returning a length-4 array of road heights [m].
        By default the result is sampled every ``dt`` seconds; pass ``t_eval``
        to sample at specific times (e.g. a lap's cumulative-time grid). For
        long lap-scale runs use ``method="LSODA"`` (auto stiff/non-stiff).
        """
        from scipy.integrate import solve_ivp

        ax_fn = ax if callable(ax) else (lambda _t, _a=float(ax): _a)
        ay_fn = ay if callable(ay) else (lambda _t, _a=float(ay): _a)
        road_fn = road if callable(road) else (lambda _t: np.zeros(4))

        y0 = np.zeros(14) if state0 is None else np.asarray(state0, float)
        if t_eval is None:
            n_steps = max(int(round(t_end / dt)) + 1, 2)
            t_eval = np.linspace(0.0, t_end, n_steps)
        sol = solve_ivp(self.derivatives, (0.0, t_end), y0, t_eval=t_eval,
                        args=(ax_fn, ay_fn, road_fn), method=method,
                        rtol=1e-6, atol=1e-8,
                        max_step=max_step if max_step is not None else dt)

        z_u = sol.y[3:7].T
        road_arr = np.array([road_fn(tt) for tt in sol.t])
        loads = np.maximum(self.p.n_static + self.p.k_t * (road_arr - z_u), 0.0)
        return RideResult(t=sol.t, z_s=sol.y[0], pitch=sol.y[1], roll=sol.y[2],
                          z_u=z_u, tire_load=loads)


if __name__ == "__main__":
    # Step-steer: 1.0g lateral applied at t=0, compare soft vs stiff dampers.
    from dataclasses import replace as _replace

    base = VehicleConfig()
    soft = RideModelParams.from_config(_replace(
        base, damper_bump_f=500, damper_rebound_f=800,
        damper_bump_r=500, damper_rebound_r=800))
    stiff = RideModelParams.from_config(_replace(
        base, damper_bump_f=4000, damper_rebound_f=7000,
        damper_bump_r=4000, damper_rebound_r=7000))

    for name, prm in (("soft", soft), ("stiff", stiff)):
        r = SevenDOFRideModel(prm).simulate(1.5, ax=0.0, ay=1.0 * G)
        roll_deg = np.degrees(r.roll)
        settle = roll_deg[-1]
        print(f"{name:5s}  steady roll={settle:5.2f} deg  "
              f"peak roll={roll_deg.max():5.2f} deg  "
              f"FL load={r.tire_load[-1,0]:6.0f} N  "
              f"FR load={r.tire_load[-1,1]:6.0f} N")
