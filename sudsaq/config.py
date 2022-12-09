"""
"""
import copy
import logging
import os
import yaml

Logger = logging.getLogger('MilkyLib/config.py')


class Null:
    """
    Null class. Used by Section to access nonexistant configuration keys
    """
    def __call__(self):
        pass

    def __deepcopy__(self, memo):
        return type(self)()

    def __bool__(self):
        return False

    def __eq__(self, other):
        if other is Null:
            return True
        if type(other) in [type(None), type(self)]:
            return True
        return False

    def __getattr__(self, key):
        return self

    def __getitem__(self, key):
        return self

    def __contains__(self):
        return False

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __repr__(self):
        return 'Null'

    def __str__(self):
        return 'Null'

    def get(self, key, other=None):
        return other

    def keys(self):
        return []

    def items(self):
        return {}

class Section:
    """
    A section of the Config, essentially acts like a dict with dot notation
    """
    def __init__(self, name=None, data={}):
        self.__dict__['_name'] = name
        self.__dict__['_data'] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = Section(key, value)
            else:
                self._data[key] = value

    def __reduce__(self):
        return (type(self), (self._name, self._data))

    def __deepcopy__(self, memo):
        new = type(self)(self._name)
        memo[id(self)] = new
        new._data = copy.deepcopy(self._data, memo)
        return new

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self._data:
            return self._data[key]
        return Null()

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        elif isinstance(value, dict):
            self._data[key] = Section(key, value)
        else:
            self._data[key] = value

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __delitem__(self, key):
        del self._data[key]

    def __repr__(self):
        sections, attributes = [], []
        for key, value in self.items():
            if isinstance(value, Section):
                sections.append(key)
            else:
                attributes.append(key)
        return f'<Section {self._name} (attributes={attributes}, sections={sections})>'

    def get(self, key, other=None):
        return self._data.get(key, other)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

class Config():
    """
    Reads a yaml file and creates a configuration from it
    """
    _data  = {}
    _flags = Section('flags', {
        'initialized': False,
        'active'     : Section(),
        'active_name': None,
        'default'    : None
    })

    @classmethod
    def __init__(cls, string=None, active=None, default='default'):
        """
        Loads a config.yaml file

        Parameters
        ----------
        string: str
            Either a path to the config.yaml file to read or a yaml string to
            parse
        active: str, optional
            Sets the active section
        default: str, optional
            Defaults to this section when a key is not found in the active
            section
        """
        if string is not None:
            # Returning from __reduce__
            if isinstance(string, tuple):
                cls._flags = string[0]
                cls._data  = string[1]
            # Otherwise initialize
            else:
                # File case
                if os.path.exists(string):
                    cls._flags.file = string
                    with open(string, 'r') as f:
                        data = yaml.load(f, Loader=yaml.FullLoader)
                # String case
                else:
                    cls._flags.file = None
                    data = yaml.load(string, Loader=yaml.FullLoader)

                if active not in data:
                    raise AttributeError(f'Active section {active!r} does not exist in the YAML: {list(data.keys())}')

                cls._data = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        cls._data[key] = Section(key, value)
                    else:
                        cls._data[key] = value

                cls._flags.active_name = active
                cls._flags.default     = default if default in data else None

                # If there is a default section, deep copy it then override it using the active section
                if cls._flags.default is not None:
                    cls._flags.active = copy.deepcopy(cls._data[cls._flags.default])
                    cls._flags.active._name = f'[active] {active}'
                    cls.update(cls._data[active], cls._flags.active)
                # Otherwise copy the active section as active
                else:
                    cls._flags.active = copy.deepcopy(cls._data[cls._flags.active_name])
                    cls._flags.active._name = f'[active] {active}'

                cls._flags.initialized = True

    @classmethod
    def __reduce__(cls):
        return cls, ((cls._flags, cls._data), )

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @classmethod
    def __setattr__(cls, key, value):
        if cls._flags.initialized:
            cls._data[key] = value
        else:
            cls._flags[key] = value

    @classmethod
    def __setitem__(cls, key, value):
        cls._data[key] = value

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        else:
            return self._flags.active[key]

    def __getitem__(self, key):
        return self.__getattr__(key)

    def get(self, key, other=None):
        value = self.__getattr__(key)
        if not isinstance(value, Null):
            return value
        return other

    def __repr__(self):
        sections, attributes = [], []
        for key, value in self._flags.active.items():
            if isinstance(value, Section):
                sections.append(key)
            else:
                attributes.append(key)
        return f'<Config(attributes={attributes}, sections={sections})>'

    def __call__(self, key=None):
        sections, attributes = [], []
        for key, value in self._data.items():
            if isinstance(value, Section):
                sections.append(key)
            else:
                attributes.append(key)
        return f'<Config _data=(attributes={attributes}, sections={sections})>'

    @classmethod
    def update(cls, a, b=None):
        """
        Updates the `b` section with the arguments of the `a` section

        Parameters
        ----------
        a: milkylib.config.Section
            Section object to be read from
        b: milkylib.config.Section, optional
            Section object to be updated
        """
        # Use the default section if not given
        if b is None:
            b = cls._flags.active

        for key, value in a.items():
            if key in b and isinstance(value, Section):
                cls.update(value, b[key])
            else:
                b[key] = value

    @classmethod
    def reset(cls, name=None):
        """
        Resets a config back to its initialized state. Can be provided a name
        to switch the active section.

        Parameters
        ----------
        name: str, optional
            The string name of the section to set active
        """
        cls._flags.initialized = False
        cls.__init__(cls._flags.file, name or cls._flags.active_name, cls._flags.default)
