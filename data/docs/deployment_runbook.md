# Service Deployment Runbook

## Overview
This document describes the standard procedure for deploying a service to an environment.

## Required Information
- Service name
- Target environment (staging or production)
- Artifact version or image tag
- Deployment method (rolling, blue-green)

## Deployment Steps
1. Verify you have deployment access to the target environment.
2. Confirm the artifact version exists in the registry.
3. Review recent changes and associated tickets.
4. Trigger the deployment using the deployment tool.
5. Monitor logs and metrics during rollout.

## Validation
- Service health checks pass.
- No error spikes in logs.
- Key endpoints respond successfully.

## Rollback
If errors are detected after deployment:
- Stop the rollout.
- Redeploy the previously stable version.
- Confirm system stability.
