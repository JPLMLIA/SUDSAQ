# SUDS Air Quality

[TODO]


## Disclaimer

This pipeline is designed for **RESEARCH** purposes. Due to its high configurability, executing it may involve dealing with considerable complexity. Understanding the intricacies of its configuration may require delving into the source code itself, as it's challenging to encapsulate every potential scenario or variation. To provide insight into its usage, we've included several [examples](...) with accompanying descriptions. Should you have additional inquiries, feel free to open a ticket in this repository or contact the maintainer via email.

## Installation and Usage

1. Clone the repo
```bash
$ git clone https://github-fn.jpl.nasa.gov/SUDS-Air-Quality/suds-air-quality.git
```
2. Install the package
```bash
$ pip install suds-air-quality/
```
3. Invoke the package
```python
import sudsaq
```
4. Use the CLI
```bash
$ sudsaq --help
```

## Getting Started

The package is primarily executed through its CLI after a [mlky](https://jammont.github.io/mlky/) configuration has been created.

### `mlky` Configurations

SUDSAQ uses [mlky]() for its configurations. Refer to the example configs under `sudsaq/configs` for examples.

## Copyright

Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.
