# SUDS Air Quality

**S**cience **U**nderstanding **D**ata **S**cience: **A**ir **Q**uality is a research project from the Machine Learning and Instrument Autonomy group at the Jet Propulsion Laboratory. This project investigates driving features of pollution on scales finer than physical models operate on. By discovering and understanding these features, scientists may use this information to further improve the performance of physical models to better predict pollution.

## Disclaimer

This pipeline is designed for **RESEARCH** purposes. Due to its high configurability, executing it may involve dealing with considerable complexity. Understanding the intricacies of its configuration may require delving into the source code itself, as it's challenging to encapsulate every potential scenario or variation. Should you have additional inquiries, feel free to open a ticket in this repository or contact the maintainer via email.

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

SUDSAQ uses [mlky](https://github.com/jammont/mlky) for its configurations. Examples can be found in `sudsaq/configs`.

`mlky` generates configurations by "patching" sections of the config file onto other sections. Patches are defined as a string delimited by `<-`. Using `sudsaq/configs/v4.yml` for reference, sections available include:

  - `default`
  - `mac`
  - `RFQ`
  - `v4`
  - `2005-20`
  - `bias-median`
  - `NorthAmerica`
  - `jan`

Some sections automatically patch another section by defining `mlky.patch` in the section. For example, `mac` auto patches `default`.

```yaml
mac:
  mlky.patch: default
```

Examples of patch strings to create working configs:

  - `mac<-dev-limited<-toar-v2-3<-bias-median<-jan`
  - `mac<-v4<-toar-v2-3<-bias-mean<-mar`
  - `mac<-v4<-RFQ<-2005-20<-bias-median<-jul`
    - Start with `mac`
    - Use the `v4` definition
    - Enable the RandomForestQuantileRegressor `RFQ`
    - Expand the range from 2005 to 2020 `2005-20`
    - Limit the tree depth `cap-depth`
    - Set the target as `bias-median` and load the necessary data files
    - Process only for `jul`

## Copyright

Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.
