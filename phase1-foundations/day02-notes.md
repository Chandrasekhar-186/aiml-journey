## Decorators — Plain English

A decorator is a function that wraps another function
to add extra behavior WITHOUT changing its code.

Real MLOps example:
@mlflow.trace → automatically logs every function 
call to MLflow experiment tracker

Syntax:
@decorator_name
def my_function():
    pass

This is identical to:
my_function = decorator_name(my_function)
