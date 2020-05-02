from . import _dll


def refresh():
    _dll.refresh_xml()


def runWithErrorCheck():
    _dll.openmc_run_with_errorcheck()
