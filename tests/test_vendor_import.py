def test_clm_public_exports():
    from clm import CLMClient, Noul, Choice, Score
    assert CLMClient is not None
    assert Noul(instructions="x").instructions == "x"

def test_console_scripts_resolvable():
    from clm.server import main as serve_main
    from clm.heads import download_main
    assert callable(serve_main) and callable(download_main)
