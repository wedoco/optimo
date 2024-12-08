FROM python:3.11

# Install packages
RUN apt update && \
    apt install --no-install-recommends -y \
    git \
    wget \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    cmake \
    xdg-utils && \
    rm -rf /var/lib/apt/lists/*

# Install OpenModelica and Modelica libraries
# Available OMC versions can be found at https://build.openmodelica.org/apt/dists/focal/
ENV OMC_VERSION 1.25.0~dev-218-gb7eae8a-1
RUN curl -fsSL http://build.openmodelica.org/apt/openmodelica.asc | gpg --dearmor -o /usr/share/keyrings/openmodelica-keyring.gpg\
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/openmodelica-keyring.gpg] https://build.openmodelica.org/apt \
    $(lsb_release -cs) nightly" | tee /etc/apt/sources.list.d/openmodelica.list > /dev/null
RUN apt update && apt install --no-install-recommends -y \
    omc=$OMC_VERSION \
    omlibrary=$OMC_VERSION \
    gdb
RUN apt clean && rm -rf /var/lib/apt/lists/*

# Add developer user with permissions in the working directory
ARG USER="developer"
RUN useradd -ms /bin/bash $USER
ENV HOME /home/$USER
RUN mkdir -m 1777 ${HOME}/optimo
ENV PYTHONPATH ${HOME}/optimo
USER $USER

# Install Poetry 
RUN pip install -U pip
RUN pip install poetry==1.6.1
ENV PATH="/home/$USER/.local/bin:$PATH" 
ENV POETRY_CACHE_DIR=${HOME}/optimo

# Configure Poetry to create virtual environments inside the project directory
RUN poetry config virtualenvs.in-project true
