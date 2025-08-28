from .mount_model import MountModel
from .mount_factory import mount_factory

from .rethink_mount import RethinkMount
from .phantom_mount import PhantomMount
from .null_mount import NullMount


MOUNT_MAPPING = {
    "RethinkMount": RethinkMount,
    "PhantomMount": PhantomMount,
    None: NullMount,
}

ALL_MOUNTS = MOUNT_MAPPING.keys()