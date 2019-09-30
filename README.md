# What's this
This repo contains collections of scripts for [rlpy](https://github.com/rlpy/rlpy)
experiments.

**WARNING**
This repository is moved to examples directory of [rlpy3](https://github.com/kngwyu/rlpy3)
and then archived.

## Requirements
- Python >= 3.6
- [rlpy3](https://github.com/kngwyu/rlpy) (My Python3 oriented fork of RLPy)
- [click](https://click.palletsprojects.com/en/7.x/)

## Setup
Install [pipenv](https://pipenv.readthedocs.io/en/latest/) and then
```bash
pipenv --site-packages --three install
```

## Example usages

```bash
pipenv run python gridworld.py --agent=ifddk-q train --visualize-performance=1
```

## Screenshots
![Gridworld](./pictures/gridworld11x11-rooms.png)

## License
This project is licensed under Apache License, Version 2.0
([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).
