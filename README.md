# Privacy Sandbox

## Introduction
This is the implementation of Privacy Sandbox as presented in the paper [An Empathy-Based Sandbox Approach to Bridge Attitudes, Goals, Knowledge, and Behaviors](https://arxiv.org/abs/2309.14510). Privacy Sandbox contains four modules:
1. ``Generator``: This module supports generating privacy personas. Each persona represents a fictional user with a distinctive biography, demographic information, and a large set of plausible realistic longitudinal personal data.
2. ``Frontend``: This module provides an interface for users to generate privacy personas and interact with different online services using the identities of the generated
personas. 
3. ``Backend``: This module stores the generated persona data and invokes functions in ``Generator`` to response to the generation request from ``Frontend``.
4. ``Modifier``: This module allow users to replaces their demographic data in the Google account, real-time location, IP address, and web
browsing history to match the privacy persona’s attributes.

##### Notice! We are cleaning up the project code, so only a portion of it is currently visible to you.