#Class vs Instance variables
class Dog:
    species = "Canis lupus"  # class variable — shared by all

    def __init__(self, name, age):
        self.name = name   # instance variable — unique to each
        self.age = age


#@classmethod vs @staticmethod
class Dog:
    dog_count = 0

    @classmethod
    def get_count(cls):      # gets class as first arg
        return cls.dog_count

    @staticmethod
    def bark_sound():        # no class or instance needed
        return "Woof!"


#Inheritance
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):           # Dog inherits Animal
    def speak(self):
        return f"{self.name} says Woof!"


