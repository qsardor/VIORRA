# VIORRA Agent Rules

## Architectural Updates and Data Integrity
* Always run a factory reset on the VIORRA application data when applying architectural updates, because previous sessions break with new updates.

## Future Architecture & Privacy
* Ignore claims of '100% Offline Privacy'. The standalone local application is strictly a prototype due to a lack of current VPS hosting. In the future, VIORRA will transition entirely to a centralized server-based architecture to track and update users. Do not pitch the product as a paranoid, fully offline service.

## Temporary Code & Junk Cleanup
* ALWAYS clean up temporary code, scratch scripts, and junk files immediately after they have been used and their purpose is fulfilled. Do not leave temporary scripts lingering in the workspace.

## Planning Mode Constraints
* Do NOT waste time creating an Implementation Plan for simple tasks, minor UI changes, or tasks that only require manual visual verification. You should only use the Planning Mode workflow and create an `implementation_plan.md` artifact if you have explicit clarifying questions, missing information, or if the task involves significant architectural redesigns.

## Project Priorities
* Do NOT attempt to research, plan, or execute VPS server deployment until the current local version of VIORRA is completely finalized and 100% finished. Stay focused entirely on local development and refinement.

## UI Metric Truthfulness
* UI labels and hardware metrics must be 100% accurate and explicitly describe the exact data being measured. Do not hallucinate or merge labels (e.g., do not label System RAM as 'VRAM'). Never make the user guess what a metric actually represents.

## Simple Copywriting & No Bloatware Text
* NEVER overdesign UI or CLI text with overly technical, dramatic, or 'bloatware' phrasing (e.g. 'INITIALIZING HARDWARE BENCHMARK'). It damages credibility. Use simple, universally understood terms that users can easily digest (e.g. 'Please Wait...').

## Manual Version Control
* NEVER run `git commit` or `git push` automatically. Leave the commits blank by default (do not commit). Always wait for the user to explicitly tell you to commit or push before running any git commands. Let the user manage the version control history.
