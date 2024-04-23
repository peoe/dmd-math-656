# Motion Detection using QR Streaming DMD

This repository contains the files for the Course MATH-656 held in March and April 2024 at EPFL.

<details open="open">
    <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
    <ol>
        <li>
            <a href="#setup">Setup</a>
            <ol>
                <li><a href="#setting-up-the-python-environment">Setting up the Python Environment</a></li>
                <li><a href="#using-the-makefile">Using the Makefile</a></li>
            </ol>
        </li>
        <li><a href="#license">License</a></li>
    </ol>
</details>

## Setup

### Setting up the Python Environment

To run the code you need a recent Python installation. It is recommended to set up a virtual environment prior to installing the required packages by running the following commands:

```zsh
python -m /path/to/your/venv/directory/your-venv-name
source /path/to/your/venv/directory/your-venv-name/bin/activate
```

Once you're in the environment where you'd like to install the packages run the following command:

```zsh
pip install -r packages
```

### Using the Makefile

To run the experiments you can either use the provided [Makefile](https://github.com/peoe/dmd-math-656/blob/main/Makefile) to automatically download all experiment data and run the scripts, or manually call each individual script.

To download the data and run the scripts simply call

```zsh
make
```

If you only want to download the data, run

```zsh
make data
```

If you only want to call the scripts, execute the `all` target of the makefile. The `all` target runs motion detection and foreground/background separation on threee datasets each, plots residuals, saves data matrices to file, and renders GIFs of the results. This will also download all the data if the data has not yet been stored locally.

```zsh
make all
```

The other possible make targets are:
 * `gifs`: Run all examples, but only render GIFs.
 * `matrices`: Run all examples, but only save data matrices to file.
 * `plots`: Run all examples, but only plot residuals.

## License

This software is distributed under the GPL-3.0 License, see [LICENSE](https://github.com/peoe/dmd-math-656/blob/main/LICENSE) for more details.

## Contact

Peter Oehme - [peter.oehme@epfl.ch](mailto:peter.oehme@epfl.ch)
