"""Repair loop for an asset whose geometric candidates all failed their trials.

``request`` turns the failed run into an evidence package plus the playbook for a diagnosis agent; ``tool`` is
the agent's ``try_grasp`` (trial grasps from the same start state); ``writeback`` re-trials the agent's repair
once and, if it lifts, writes it to the asset (``robotsmith.grasp.annotations``) for the next run to use.
"""
