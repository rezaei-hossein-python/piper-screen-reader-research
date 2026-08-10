def test_phase2al_freezes_a5_family_definition():
    from phase2aj_policy import POLICY_FAMILIES
    assert POLICY_FAMILIES["a5"] == {"E1","E2","E3"}
