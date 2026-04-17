# scenarios/__init__.py
"""
Scenario modules for the queueing lecture.

Conventions (each scenario module should expose):
  - build_cfg() -> dict
  - make_baselines() -> dict with keys:
        "policies": Dict[str, Dict[str, RouteDecisionPolicy]]   (routing DP policies)
        "dispatch": Optional[Dict[str, Dict[str, DispatchPolicy]]]  (optional per-policy dispatch maps)
        "baseline_name": str
        "T_des": float
        "T_env": float
        "n_rep": int
  - run_des_demo()
  - run_crn_demo()
"""