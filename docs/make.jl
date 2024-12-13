# Documenter creates html files in build directory. 
# To generate the documentation: navigate (set working directory) to the docs directory, then run make.jl with Julia.

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

# NB: The following code cannot run on my local machine. It has to be on 
# GitHub Actions, where you set up a CI service to automatically run your make.jl script whenever you push changes to your repository
# Perhaps I just need the repo to be public? 
#  THE .github/workflows IS MISSING !!! It's there in smartboost .... 
# " Explain in detail how to set up the CI service in GtHub Actions."

#=
deploydocs(repo="github.com/PaoloGiordani/HTBoost.jl.git",
    target="build",
    branch = "gh-pages",
    devbranch="main",
    push_preview = false,
    )

=#

# Pages: NB they are public !!! 
#  https://paologiordani.github.io/HTBoost.jl
#  Generate and deploy your documentation: Run the make.jl script to generate and 
#  deploy your documentation. You can do this by running the following command in the docs/ directory:
#  julia make.jl


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
