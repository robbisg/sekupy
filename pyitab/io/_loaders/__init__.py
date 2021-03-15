from ._hcp import load_hcp_motor, load_hcp_blp
from ._reftep import load_reftep_iplv, load_reftep_power, \
                     load_reftep_sensor


mambo_mapper = {
    'hcp-motor': load_hcp_motor,
    'reftep-iplv': load_reftep_iplv,
    'reftep-power': load_reftep_power,
    'reftep-sensor': load_reftep_sensor,
    'hcp-blp': load_hcp_blp
}