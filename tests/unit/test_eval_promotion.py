from ups.eval.promotion import (
    evaluate_promotion_rules,
    parse_promotion_rule,
    promotion_rules_from_config,
)


def test_parse_promotion_rule():
    rule = parse_promotion_rule("decoded_rollout_nrmse<=0.2")
    assert rule.metric == "decoded_rollout_nrmse"
    assert rule.operator == "<="
    assert rule.threshold == 0.2


def test_parse_promotion_rule_with_reducer():
    rule = parse_promotion_rule("max:family_*_decoded_rollout_nrmse<=0.4")
    assert rule.metric == "family_*_decoded_rollout_nrmse"
    assert rule.operator == "<="
    assert rule.threshold == 0.4
    assert rule.reducer == "max"


def test_evaluate_promotion_rules_pass_and_fail():
    rules = [
        parse_promotion_rule("decoded_rollout_nrmse<=0.2"),
        parse_promotion_rule("transfer_decoded_rollout_nrmse<=0.3"),
    ]
    result = evaluate_promotion_rules(
        {
            "decoded_rollout_nrmse": 0.15,
            "transfer_decoded_rollout_nrmse": 0.35,
        },
        rules,
    )
    assert not result.passed
    assert len(result.failed_rules) == 1
    assert result.failed_rules[0].startswith("transfer_decoded_rollout_nrmse<=0.3")


def test_promotion_rules_from_config_supports_dict_and_string():
    rules = promotion_rules_from_config(
        {
            "evaluation": {
                "promotion": {
                    "rules": [
                        "decoded_rollout_nrmse<=0.2",
                        {
                            "metric": "transfer_decoded_rollout_nrmse",
                            "operator": "<=",
                            "threshold": 0.3,
                            "label": "transfer gate",
                        },
                    ]
                }
            }
        }
    )
    assert len(rules) == 2
    assert rules[0].metric == "decoded_rollout_nrmse"
    assert rules[1].label == "transfer gate"


def test_evaluate_promotion_rules_supports_wildcard_reducers():
    rules = [
        parse_promotion_rule("max:family_*_decoded_rollout_nrmse<=0.2"),
        parse_promotion_rule("mean:task_*_decoded_step1_nrmse<=0.15"),
    ]
    result = evaluate_promotion_rules(
        {
            "family_conservation_decoded_rollout_nrmse": 0.18,
            "family_transport_decoded_rollout_nrmse": 0.22,
            "task_burgers1d_decoded_step1_nrmse": 0.1,
            "task_advection1d_decoded_step1_nrmse": 0.12,
        },
        rules,
    )
    assert not result.passed
    assert len(result.failed_rules) == 1
    assert result.failed_rules[0].startswith("max:family_*_decoded_rollout_nrmse<=0.2")


def test_evaluate_promotion_rules_reports_missing_wildcard_metric_group():
    rules = [parse_promotion_rule("max:family_*_decoded_rollout_nrmse<=0.2")]
    result = evaluate_promotion_rules({"decoded_rollout_nrmse": 0.1}, rules)
    assert not result.passed
    assert result.missing_metrics == ["max:family_*_decoded_rollout_nrmse"]
