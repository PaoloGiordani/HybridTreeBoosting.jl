# This file creates the scripts that generate the documentation. 
# To generate the documentation: navigate to the docs directory, then run make.jl with Julia.
# You can do this by navigating to the docs directory in your terminal and running julia make.jl.
# you can optionally host your documentation on a website like GitHub Pages or Read the Docs.

push!(LOAD_PATH, "../src/")

using Documenter, HTBoost

# in this case examples.md must be in the docs/src directory, or otherwise specify the path. 
pages=[
    "Introduction" => "../README.md",           
    "Parameters" => "parameters.md",                          
    "API" => "JuliaAPI.md",
    "Tutorials" => "src/Tutorials.md",
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
deploydocs(repo="github.com/HTBoost/HTBoost.jl.git",
    target="build",
    devbranch="main")
=#


#=


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




=#
