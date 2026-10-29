# RoboSmith

English | [中文](README.zh-CN.md)

**Synthetic data generation for robot manipulation on AMD ROCm.**

RoboSmith turns assets plus a declarative task into recorded expert episodes. You describe what is on the table,
which steps to run, and what counts as success in a few lines of Python; RoboSmith builds the scene in Genesis,
plans grasps and collision-aware motion, rolls out the episodes, and exports the successful ones as a LeRobot
dataset with videos. It is data infrastructure, not a policy-training framework.

<table align="center">
  <tr>
    <td width="50%">
      <a href="videos/pick_place_into_drawer_ep000_up.mp4"><img src="videos/pick_place_into_drawer_ep000_up.gif" width="100%"></a>
    </td>
    <td width="50%">
      <a href="videos/pick_place_onto_supporter_ep000_up.mp4"><img src="videos/pick_place_onto_supporter_ep000_up.gif" width="100%"></a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Drawer · long-horizon articulated</b><br>open → pick → place → close</td>
    <td align="center"><b>Two-tier supporter · side insertion</b><br>pick from the upper shelf, insert into the lower one</td>
  </tr>
</table>

## What it generates

| Asset kind | Actions | Success predicates |
|---|---|---|
| rigid | `pick`, `place` (geometric grasp candidates, tried in simulation) | `inside`, `on`, `lifted`, `stacked`, `aligned` |
| articulated | `open`, `close` (prismatic joints) | `joint_opened`, `joint_closed` |

A task is a short Python file: scene, skill intents, and a success tree. See
[`scenarios/pick_place_into_drawer.yaml`](scenarios/pick_place_into_drawer.yaml).

## Get started

- [Quick start](docs/quick-start.md): from a fresh GPU node to a recorded drawer episode.
- [Design](docs/design.md): teacher and student, data flow, modules, execution, verification, and the projects
  RoboSmith builds on.
- [Writing scenarios](scenarios/README.md).
- Blog post: [RoboSmith — a ROCm-native synthetic-data pipeline for embodied interactive manipulation](https://andyluo7.github.io/rocm/amd/mi300x/robotics/embodiedai/sdg/2026/07/01/robosmith-rocm-native-synthetic-data-pipeline/).

Runs on MI300 (gfx942) and MI350 (gfx950). Motion planning comes from [rocRobo](https://github.com/ZJLi2013/rocRobo),
run as a sidecar container.

## Status

[v0.1.0](CHANGELOG.md) is a scripted-teacher data generator. The effect of its data on policy training has not been
validated: there is no closed-loop evaluation entry yet. Main limits:

- Contact segments are unchecked straight lines, and a failed `plan_motion` falls back to one.
- Articulation supports prismatic joints only.
- One scene per process.

## Contributing

All changes land through pull requests; see [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and [AI_POLICY.md](AI_POLICY.md). Report vulnerabilities privately per [SECURITY.md](SECURITY.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
