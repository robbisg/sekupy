from ._hcp import load_hcp_motor, load_hcp_blp
from ._reftep import load_reftep_iplv, load_reftep_power, \
                     load_reftep_sensor, load_reftep_conn


def mambo_mapper(method):
    
    mapper = {
        'hcp-motor': load_hcp_motor,
        'reftep-iplv': load_reftep_iplv,
        'reftep-power': load_reftep_power,
        'reftep-sensor': load_reftep_sensor,
        'reftep-conn': load_reftep_conn,
        'hcp-blp': load_hcp_blp
    }

    return mapper[method]