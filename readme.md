<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="./resources/readme_images/logo.png" alt="Logo" height="80">
  </a>

  <h3 align="center">
    Application and analysis of Curiosity Driven Reinforcement Learning to solve the job shop problem in the context of industrial manufacturing
  </h3>

  <p align="center">
    Master Thesis by Alexander Nasuta
    <br />
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#project-strructure">Project Structure</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#hardware">Hardware</a></li>
        <li><a href="#python-environment-management">Python Environment Management</a></li>
        <ul>
            <li><a href="#mac">Mac</a></li>
            <li><a href="#windows">Windows</a></li>
        </ul>
        <li><a href="#idea">IDEA</a></li>
        <ul>
            <li><a href="#PyCharm Setup">PyCharm Setup</a></li>
        </ul>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project
This Master Thesis tries to apply [Curiosity-driven Reinforcement Learning](https://gym.openai.com/) 
to the Job Shop Scheduling Problem (JSP).

![Enviorment Screenshot][screenshot]


This project contains all that code, that was used in the thesis. It includes a custom 
[OpenAi Gym Environment](https://gym.openai.com/) and a some reinforcement learning algorithms, that were 
evaluated on that environment.

### Project Structure
This project ist structured according to [James Murphy's testing guide](https://www.youtube.com/watch?v=DhUpxWjOhME)

### Built With

This project uses (among others) the following libraries

* [OpenAI Gym](https://gym.openai.com/)
<!-- todo: Add all major libs-->

## Getting Started

In this Section describes the used Setup and Development tools.

### Hardware

All the code was developed and tested on an Apple M1 Max 16" MacBook Pro (16-inch, 2021) with 64 GB Unified Memory.

The **code** should run perfectly fine on other devices and operating Systems. 

Only the **tools** on Macs require slightly 
a slightly different setup than the alternatives on other operating systems. 


### Python Environment Management

#### Mac
On a Mac I recommend using [Miniforge](https://github.com/conda-forge/miniforge) instead of more common virtual
environment solutions like [Anacond](https://www.anaconda.com) or [Conda-Forge](https://conda-forge.org/#page-top).

Accelerate training of machine learning models with TensorFlow on a Mac requires a special installation procedure, 
that can be found [here](https://developer.apple.com/metal/tensorflow-plugin/).

Setting up Miniforge can be a bit tricky (especially when Anaconda is already installed).
I found this [guide](https://www.youtube.com/watch?v=w2qlou7n7MA) by Jeff Heaton quite helpful.

#### Windows

On a **Windows** Machine I recommend [Anacond](https://www.anaconda.com), since [Anacond](https://www.anaconda.com) and 
[Pycharm](https://www.jetbrains.com/de-de/pycharm/) are designed to work well with each 
other. 

### IDEA

I recommend to use [Pycharm](https://www.jetbrains.com/de-de/pycharm/).
Of course any code editor can be used instead (like [VS code](https://code.visualstudio.com/) 
or [Vim](https://github.com/vim/vim)).

This Section goes over a few recommended step for setting up the Project properly inside [Pycharm](https://www.jetbrains.com/de-de/pycharm/).

#### PyCharm Setup
1. Mark the `src` directory as `Source Root`.
```
   right click on the 'src' -> 'Mark directory as' -> `Source Root`
```

2. Mark the `resources` directory as `Resource Root`.
```
   right click on the 'resources' -> 'Mark directory as' -> `Resource Root`
```

3. (optional) Mark the `tests` directory as `Test Source Root`.
```
   right click on the 'tests' -> 'Mark directory as' -> `Test Source Root`
```

4. (optional) When running a script enable `Emulate terminal in output console`
```
Run (drop down) | Edit Configurations... | Configuration | ☑️ Emulate terminal in output console
```

### Usage

To run this Project locally on your machine follow the following steps:

1. Clone the repo
   ```sh
   git clone todo
   ```
2. Install the python requirements packages
    ```sh
   pip install -r ./requirements.txt
   ```
   
In order to be able to run **tests** you need to follow this additional tests: 

3. (optional) Install the modules of the project. For more info have a look at 
[James Murphy's testing guide](https://www.youtube.com/watch?v=DhUpxWjOhME)
    ```sh
   pip install -e .
   ```
    
   ```sh
   pip install -r ./requirements_dev.txt
   ```

## Testing

For testing make sure that the dev dependencies are installed (`./requirements_dev.txt`) and the models of this 
project are set up (i.e. you have run `pip install -e .`).  

Then you should be able to run

```sh
mypy src
```

```sh
flake8 src
```

```sh
pytest
```

or everthing at once using `tox`

```sh
tox
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- MARKDOWN LINKS & IMAGES todo: add Github, Linked in etc.-->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[screenshot]: resources/readme_images/screenshot.png


