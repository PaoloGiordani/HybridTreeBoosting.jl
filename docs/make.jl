# This file creates the scripts that generate the documentation. 
# NB: Documentation should not be pushed to the Master directory, but to the gh-pages

# To generate the documentation: navigate to the docs directory, then run make.jl with Julia.
# You can do this by navigating to the docs directory in your terminal and running julia make.jl.
# PG: may need to be the terminal. See Documenter.jl

# you can host your documentation on a website like GitHub Pages or Read the Docs.
push!(LOAD_PATH, "../src/")

using Documenter, HTBoost

# EXAMPLE FOR PAGES 
# NB: in this case examples.md must be in the docs/src directory, or otherwise specify the path. 
pages=[
    "Introduction" => "../README.md",           
    "Parameters" => "parameters.md",                          
    "API" => "API.md",
    "Tutorials" => "Tutorials.md",
    "Examples (julia scripts)" => "../examples/examples.md",
    #"Table of Contents" => "toc.md",
]

makedocs(
    sitename="HTBoost.jl",
    authors = "Paolo Giordani",
    modules=[HTBoost],
    format=Documenter.HTML(
        sidebar_sitename="false", # able or disable the site name on the site bar
        edit_link = "main"        # adds an "Edit on GitHub" button on documentation  
      ),
    pages=pages,
    repo = "https://github.com/PaoloGiordani/HTBoost.jl", # link for edit_link
    assets = ["assets"],     # directories for images, css, etc. that are copied to the generated documentation site
)  

    
#=
# Is this related to the page for the documentation I need to set up with GitHub Pages?
# ????? 
deploydocs(repo="github.com/Evovest/EvoTrees.jl.git",
    target="build",
    devbranch="main")
=#


#=

# How can I include code examples in my Julia package documentation using Documenter.jl?

#=

To include code examples in your Julia package documentation using Documenter.jl,
you can use the @example or @repl macros in your markdown files. Here's how you can do it:

1) @example Macro: The @example macro allows you to include a block of Julia code that
will be evaluated when the documentation is built. The output of the code will be included in the documentation.
Here's an example:


```@example
using HTBoost
# Some example usage of your package (actual code goes here)
```

2) **@repl Macro**: The `@repl` macro is similar to `@example`, but it also includes the input code in the documentation, making it look like a REPL session. Here's an example:

```markdown
```@repl
using HTBoost
# Some example usage of your package

3) 
**Including Files**: If you have longer examples that are stored in separate files, you can include them using the `@include` macro. For example, if you have an example in a file called `example.jl`, you can include it like this:

```markdown
```@include example.jl


TO GENERATE A TABLE OF CONTENTS FOR MY PACKAGE DOCUMENTATION USING Documenter.jl

To generate a table of contents (TOC) for your Julia package documentation using Documenter.jl,'
you can use the @contents macro. Here's how you can do it:

1) Create a markdown file for your TOC: In your docs/src directory, create a new markdown file for your TOC.
You might name it toc.md.

2) Add the @contents macro to your TOC file: In your toc.md file, add the @contents macro where you want your TOC to appear. Here's an example:

    # Table of Contents

```@contents


3. **Add your TOC file to your documentation**: In your `make.jl` file, add your `toc.md` file to the `pages` argument of the `makedocs` function. Here's an example:

```julia
makedocs(
    sitename = "HTBoost.jl",
    modules = [HTBoost],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "Table of Contents" => "toc.md",
        "API" => "api.md",
        "Examples" => "examples.md",
    ],
    repo = "https://github.com/yourusername/HTBoost.jl/blob/{commit}{path}#L{line}",
    assets = ["assets"],
)


The @contents macro will automatically generate a TOC based on the headers in your markdown files. The TOC will include links to each section, allowing readers to easily navigate your documentation.


LINKS TO EXTERNAL WEBSITES 

Standard markdown syntax:
[Link text](https://www.example.com)

For example, if you want to create a link to the Julia documentation, you could do it like this:
Check out the [Julia documentation](https://docs.julialang.org/)
This will create a link that says "Julia documentation" and points to https://docs.julialang.org/.



=#

=#