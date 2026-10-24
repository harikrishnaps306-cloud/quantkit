#!/usr/bin/env python3
"""Fail unless the GitHub deployment environment is protected as the release expects.

    python .github/scripts/check_environment.py OWNER/REPO ENVIRONMENT [--branch master]

Reads the environment through the GitHub REST API (GITHUB_TOKEN is used when set;
the repository is public, so the read also works without it) and exits 1 unless:

* the environment exists (GitHub silently creates a missing environment, with no
  protection, the first time a job names it, which is exactly what this guards
  against);
* a required-reviewers rule with at least one reviewer is configured;
* administrators cannot bypass the protection rules;
* deployments are restricted to custom branch rules, and the only rule is the
  expected branch.

A workflow cannot prove that a person will review; it can prove the rule that
forces a review exists at the time the check runs.
"""
import json
import os
import sys
import urllib.error
import urllib.request


def get(url):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "quantkit-env-check"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo, env = args[0], args[1]
    branch = "master"
    if "--branch" in sys.argv:
        branch = sys.argv[sys.argv.index("--branch") + 1]
    base = f"https://api.github.com/repos/{repo}/environments/{env}"
    problems = []
    try:
        data = get(base)
    except urllib.error.HTTPError as e:
        print(f"cannot read environment '{env}' of {repo}: HTTP {e.code}")
        return 1
    reviewers = [
        x["reviewer"].get("login", "?")
        for r in data.get("protection_rules", []) if r.get("type") == "required_reviewers"
        for x in r.get("reviewers", [])
    ]
    if not reviewers:
        problems.append("no required reviewers are configured")
    if data.get("can_admins_bypass") is not False:
        problems.append("administrators can bypass the protection rules")
    policy = data.get("deployment_branch_policy") or {}
    if not policy.get("custom_branch_policies"):
        problems.append("deployments are not restricted to custom branch rules")
    else:
        names = [(p["name"], p["type"]) for p in get(base + "/deployment-branch-policies").get("branch_policies", [])]
        if names != [(branch, "branch")]:
            problems.append(f"deployment branch rules are {names}, expected only [('{branch}', 'branch')]")
    print(f"environment {env}: reviewers={reviewers} can_admins_bypass={data.get('can_admins_bypass')} policy={policy}")
    if problems:
        for p in problems:
            print("FAIL:", p)
        return 1
    print("environment protection is as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
