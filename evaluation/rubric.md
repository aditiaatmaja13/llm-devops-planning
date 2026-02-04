# DevOps Plan Evaluation Rubric (v1)

For each generated plan, evaluate whether it includes the following:

## Required Information
- [ ] Target environment is explicitly stated or requested
- [ ] Service / system name is stated or requested
- [ ] Version / artifact / commit is stated or requested

## Preconditions
- [ ] Access/permissions requirements mentioned
- [ ] Dependencies or health checks mentioned
- [ ] Change window / approvals (if relevant) mentioned

## Execution Steps
- [ ] Steps are ordered
- [ ] Steps are actionable (not vague)
- [ ] Includes safety steps (e.g., backups, drain traffic) where relevant

## Validation Steps
- [ ] Clear success criteria
- [ ] Mentions logs/metrics/health checks
- [ ] Mentions user-facing verification if relevant

## Rollback Plan
- [ ] Defines rollback trigger conditions
- [ ] Provides rollback steps

## Assumptions and Risks
- [ ] Missing details are surfaced as questions
- [ ] Risks are explicitly stated (downtime, data loss, etc.)
