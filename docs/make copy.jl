#No error message when:
#- followed the steps in Documenter.jl to create the build folder. this was probably essential!!! 
#- I run the make.jl file in the docs directory (environment is docs, directory is docs)
#- I have removed HTBoost from environment. I don't think this is essential, but then using HTBoost is certain to take the local HTBoost.jl package
#- It has now created a bunch of html files in build directory. To see them ...  
# 
# I have not yet deployed the documentation on GitHub Pages. I will do this later.
# To see it, open the index.html file in the build directory from FileExplorer. 

# This file creates the scripts that generate the documentation. 
# To generate the documentation: navigate to the docs directory, then run make.jl with Julia.
# You can do this by navigating to the docs directory in your terminal and running julia make.jl.
# you can optionally host your documentation on a website like GitHub Pages or Read the Docs.


push!(LOAD_PATH, "../src/")   # for API

using Documenter, HTBoost

# in this case examples.md must be in the docs/src directory, or otherwise specify the path. 
pages=[
    "Introduction" => "index.md",           
    "Parameters" => "Parameters.md",                          
    "API" => "JuliaAPI.md",
    "Tutorials" => "Tutorials.md",
    "Examples (julia scripts)" => "Examples.md",
    #"Table of Contents" => "toc.md",
]

makedocs(
    sitename="HTBoost.jl",
    authors = "Paolo Giordani",
    modules=[HTBoost],
    format=Documenter.HTML(
        sidebar_sitename=true, # able or disable the site name on the side bar
        repolink = "https://github.com/PaoloGiordani/HTBoost.jl", # explicitly set the remote URL
        edit_link = "main"        # adds an "Edit on GitHub" button on documentation  
      ),
    pages=pages,
    repo = "https://github.com/PaoloGiordani/HTBoost.jl.git", # link for edit_link
)

# format = ... (,assets = ["assets/styles.css"]  # if you have a css file in the assets directory (style)
# logo = "assets/logo.png"       # if you have a logo in the assets directory

# Idea for symbol ..... 
# https://www.theirishroadtrip.com/celtic-tree-of-life-symbol/
# https://www.pinterest.com/pin/tree-of-life--89368373833333877/

# Uncomment the following lines to deploy the documentation on GitHub Page.
# NB: the documentation will be public !!!

deploydocs(repo="github.com/PaoloGiordani/HTBoost.jl.git",
    target="build",
    branch = "gh-pages",
    devbranch="main",
    push_preview = false,
    )


# Pages: NB they are public !!! 
#  https://paologiordani.github.io/HTBoost.jl
#  Generate and deploy your documentation: Run the make.jl script to generate and 
#  deploy your documentation. You can do this by running the following command in the docs/ directory:
#  julia make.jl

#=
To run the make.jl file
- Open the terminar (Command prompt)
- cd C:\\Users\\A1810185\\OneDrive - BI Norwegian Business School (BIEDU)\\Documents\\A_Julia-scripts\\HTBoost.jl\docs
- Activate the docs environment and run make.jl with Julia 
  julia --project=. make.jl

=#

# NB: Documenter expects all However, Documenter.jl expects all source files to be located in the src/ directory or its subdirectories.

# If I need help, use Copilot with question: "I want the documentation for my package HTBoost in GitHub pages. What do I need to do?"
# NB: after I run the make.jl above,  
# your documentation should be available at https://paologiordani.github.io/HTBoost.jl/
# This will be public, so I should do it only when ready ... 

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
