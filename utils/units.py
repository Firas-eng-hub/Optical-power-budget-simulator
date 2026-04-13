import math


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10)


def mw_to_dbm(mw: float) -> float:
    if mw <= 0:
        return float('-inf')
    return 10 * math.log10(mw)


def db_to_linear(db: float) -> float:
    return 10 ** (db / 10)


def nm_to_um(nm: float) -> float:
    return nm / 1000.0


def um_to_nm(um: float) -> float:
    return um * 1000.0
