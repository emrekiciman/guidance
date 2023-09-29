# import functions that can be called directly
from ._role import role, system, assistant, user, function
from ._gen import gen, gen_with_tools, gen_substring
from ._select import select
from ._constrain import constrain, Constraint
from . import constraints

# import the rest just to get them executed and attached as methods of the LM objects
from ._block import block as _block
from ._silent import silent as _silent
from ._append import append as _append
