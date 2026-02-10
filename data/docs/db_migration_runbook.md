# Database Migration Runbook

## Overview
This document outlines steps for deploying changes that include database migrations.

## Preconditions
- Database backups are taken.
- Migration scripts are reviewed and approved.
- Downtime requirements are understood.

## Migration Steps
1. Apply migrations in a non-production environment.
2. Verify schema changes and data integrity.
3. Deploy application changes.
4. Apply migrations in production during a low-traffic window.

## Rollback
- Restore from backup if migration fails.
- Revert application to previous version.
