# __main__.py

import click

from . import models_cli


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def main():
    '''
    \b
    riddl
    -----

    Robust Infrasound Detection via Deep Learning (RIDDL)

    Python-based tools for AI/ML-based signal analysis of infrasound data.

    '''
    pass

# Add first branch of options
@click.group('models', short_help="Build, evaluate, and use models", context_settings={'help_option_names': ['-h', '--help']})
def models():
    '''
    riddl models - AI/ML models for infrasound signal analysis
    '''
    pass 


@click.group('data', short_help="Methods to create and manipulate data", context_settings={'help_option_names': ['-h', '--help']})
def data():
    '''
    riddl data - Generate and manipulate infrasound data
    '''
    pass 


@click.group('utils', short_help="Utility tools", context_settings={'help_option_names': ['-h', '--help']})
def utils():
    '''
    riddl utils - Utility functions
    '''
    pass 

main.add_command(models)
main.add_command(data)
main.add_command(utils)

# Add model option branches
@click.group('fk', short_help="Utility tools", context_settings={'help_option_names': ['-h', '--help']})
def fk():
    '''
    riddl models fk - Models for detection and categorization of fk results.
    '''
    pass 

models.add_command(fk)

fk.add_command(models_cli.build)
fk.add_command(models_cli.train)
fk.add_command(models_cli.detect)
fk.add_command(models_cli.evaluate)

# Add model option branches
@click.group('synthetic', short_help="Generate synthetic data", context_settings={'help_option_names': ['-h', '--help']})
def synthetic():
    '''
    riddl data synthetic - Generate synthetic infrasound waveforms
    
    '''
    pass 

data.add_command(synthetic)



if __name__ == '__main__':
    main()

