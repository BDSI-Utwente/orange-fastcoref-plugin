# Orange FastCoRef Plugin

A FastCoRef coreference resolution plugin for [Orange3](https://orange.biolab.si/).

## Features

- Coreference resolution using [fastcoref](https://github.com/shon-otmazgin/fastcoref) and LingMess models
- Integration with Orange3 workflows
- Supports English and Dutch spaCy models

## Installation

Due to various conflicting dependencies and requirements, this plugin requires Python 3.11 specifically (python>=3.11, <3.12).

Install via pip:

```sh
pip install orange-fastcoref-plugin
```

Or from the latest development snapshot: 

```sh
pip install git+https://github.com/bdsi-utwente/orange-fastcoref-plugin.git
```

## Usage

After installation, launch Orange3. 

```
orange-canvas
```

or 

```
python -m Oranve.canvas
```

The "Coreference Resolution" widget will be available for use in your workflows.

## License

GPL 3.0 or later

## Links

- [Homepage](https://github.com/bdsi-utwente/orange-fastcoref-plugin)
- [Issues](https://github.com/bdsi-utwente/orange-fastcoref-plugin/issues)