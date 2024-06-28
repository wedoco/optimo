from OMPython import OMCSessionZMQ
from pathlib import Path

fmu_version = 2.0


def loadFiles(omc, mo_files=[]):
    """
    Load required Modelica files and packages.

    :param mo_files: The list of paths to the involved Modelica models and libraries.

    """

    # Load needed model files and libraries.
    for f in mo_files:
        print('Loading {} ...'.format(f))
        omc.sendExpression('loadFile(\"{}\")'.format(f))

    print('List of defined Modelica class names: {}'.format(omc.sendExpression("getClassNames()")))

def buildModelFMU(omc, mo_class, commandLineOptions=None):
    """
    Compile an FMU from a Modelica model.

    :param mo_class: The Modelica class name to be compiled.
    :return: Full path of the generated FMU.

    """

    ## set commandLineOptions if provided by users
    if commandLineOptions is not None:
        exp = "".join(["setCommandLineOptions(", "\"", commandLineOptions, "\"", ")"])
        print(exp)
        # self.omc.sendExpression('setCommandLineOptions(\"+g=Modelica\")')
        print(omc.sendExpression(exp))

    # Translate model to FMU.
    fmu_path = omc.sendExpression('buildModelFMU({0}, version=\"{1}\")'.format(mo_class, fmu_version))
    # fmu_path = self.omc.sendExpression('buildModelFMU({0}, version=\"{1}\", fmuType=\"cs\")'.format(mo_class, self.fmu_version))

    return fmu_path


model='vdp'
omc = OMCSessionZMQ()
loadFiles(omc, mo_files=[f'{model}.mo'])
fmu_path = buildModelFMU(omc, f'{model}')
fmu_path = Path(f'{model}.fmu').resolve()
