# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Abstract class for Jittable objects."""

import abc
import jax


class Jittable(metaclass=abc.ABCMeta):
    """ABC that can be passed as an arg to a jitted fn, with readable state."""

    def __new__(cls, *args, **kwargs):
        try:
            registered_cls = jax.tree_util.register_pytree_node_class(cls)
        except ValueError:
            registered_cls = cls  # already registered
        instance = super(Jittable, cls).__new__(registered_cls)
        instance._args = args
        instance._kwargs = kwargs
        return instance

    def tree_flatten(self):
        return ((), ((self._args, self._kwargs), self.__dict__))

    @classmethod
    def tree_unflatten(cls, aux_data, _):
        (args, kwargs), state_dict = aux_data
        obj = cls(*args, **kwargs)
        obj.__dict__ = state_dict
        return obj
