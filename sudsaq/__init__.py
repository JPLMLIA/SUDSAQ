try:
    # Use the new version if it's installed 
    from mlky import (
        Config,
        Null,
        Section
    )
except:
    from .config import (
        Config,
        Null,
        Section
    )
