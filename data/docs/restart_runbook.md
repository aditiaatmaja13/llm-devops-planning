# Service Restart Runbook

## Overview
This document describes how to safely restart a service.

## Preconditions
- Confirm restart will not impact active users.
- Notify stakeholders if needed.

## Restart Steps
1. Gracefully stop the service.
2. Restart the service using the service manager.
3. Monitor startup logs.

## Validation
- Service starts successfully.
- Health checks pass.
