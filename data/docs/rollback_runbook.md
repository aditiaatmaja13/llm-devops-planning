# Rollback Runbook

## Overview
This document describes how to roll back a service deployment.

## When to Roll Back
- Service health checks fail.
- Error rates exceed acceptable thresholds.
- Critical functionality is broken.

## Rollback Steps
1. Identify the last known stable version.
2. Redeploy the stable version to the affected environment.
3. Monitor logs and metrics to confirm recovery.

## Validation
- Health checks pass.
- Errors return to baseline.
