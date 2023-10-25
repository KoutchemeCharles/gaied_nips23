import ast
from python_minifier import minify
from astor import to_source
from warnings import warn

def yield_variables(root):
    """
    Retrieve all variables in the code represented by
    the given Abstract Syntax Tree.
    """
    for node in ast.walk(root):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.FunctionDef):
            yield node.name

def get_variables(code_ast):
    """ Retrieve all the variables names in a sample code. """
    return set(list(yield_variables(code_ast)))

def get_normalized_ast_representation(code):
    """ Obtain the flattened representation of the code
    with the variables names renamed. 
    """
    
    root = get_ast(code)
    if root is None: return ""
    variables = get_variables(root)
    new_var_name = {var: f"x_{i}" for i, var in enumerate(variables)}
    dumped = ast.dump(root)
    for var in variables:
        dumped = dumped.replace(f"'{var}'", f"'{new_var_name[var]}'")
    return dumped


def get_bytecode_representation(code):
    # does not work well for distinguishing code 
    return compile(code, "test.py", "exec").co_code

def keep_unique_solutions(df, code_co, fname_col, repre=get_normalized_ast_representation):
    """ Remove duplicate solutions in terms of a metric. """

    df["normalized"] = df[code_co].apply(repre)
    
    def add_representative(sub_df):
        """ Add the representative of the codes having the same appraoch. """
        if not sub_df.empty:
            sub_df["representative"] = sub_df[code_co].value_counts().index[0]
        else:
            sub_df["representative"] = sub_df[code_co]
        return sub_df
        
    groups = df.groupby([fname_col, "normalized"], as_index=False)
    new_df = groups.apply(add_representative)

    # Now, select only one of the codes which match the representative 
    new_df = new_df[new_df.representative == new_df[code_co]]
    new_df = new_df.drop_duplicates("representative", ignore_index=True, keep='last')

    return new_df # Dataset.from_pandas(new_df) 

def does_compile(code):
    return get_ast(code) is not None

def get_ast(code):
    try:
        return ast.parse(code)
    except:
        return None 
    


def clean_code(code, **args):
    """ Minify and clean a source code.
    
    Remove comments, initial docstrings, and empty blank lines. 
    Additionally, add a new docstring to the code.
    """

    code = minify(code,
                  hoist_literals=False,
                  remove_literal_statements=False,
                  remove_explicit_return_none=False,
                  convert_posargs_to_args=False,
                  rename_locals=False,
                  **args)
    return code


def simple_clean(s):
    try:
        return to_source(ast.parse(s))
    except:
        warn(f"Could not clean code {s}")
        return s