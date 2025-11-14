# Manipulation Engine - Calligraphy

## Install with a virtualenv
```bash
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip
git clone https://github.com/gabearod2/calligraphy.git
cd mengine
pip3 install -e .
```

## Testing
```bash
cd testing
python3 track_trajectory.py
```