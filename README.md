# Fuka
First Universal Kommon Ancestor (FUKA)

(k is the first mutation)

~~යසස්~

(10-ජූලි-2025)

fuka/
├─ app.py                # UI only (unchanged public API)
├─ core/
│  ├─ config.py          # dataclasses + default_config + make_config_from_dict
│  ├─ engine.py          # simulation loop; stable entry point used by app
│  ├─ history.py         # History buffers + simple logging helpers
│  ├─ env.py             # environment sources + field builder
│  ├─ physics.py         # local update ops (diffuse/decay/pump/motor/gate)
│  ├─ organism.py        # “organism” state orchestration (boundary offset etc.)
│  ├─ connections.py     # generalized connection rules (kernel/plugins)
│  └─ registry.py        # lightweight plugin registry + typed messages
└─ plugins/
   ├─ env_moving_peak.py     # example env source
   ├─ conn_gate_kernel.py    # example general connection (your current kernel)
   └─ phys_basic.py          # example physics operator bundle