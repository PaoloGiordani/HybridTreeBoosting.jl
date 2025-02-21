# Documenter creates html files in build directory. 
# Instructions: set working directory to the docs directory, then run the make.jl file in Julia.
# This produces html files that can be navigated as any other. 

push!(LOAD_PATH, "../src/")   # for API

using Documenter
using HybridTreeBoosting

# These files must be in the docs/src directory, or otherwise specify the path. 
pages=[
    "Introduction" => "index.md",           
    "Parameters" => "Parameters.md",                          
    "API" => "JuliaAPI.md",
    "Tutorials (Julia)" => "Tutorials.md",
    "Examples (Julia)" => "Examples.md",
    "Tutorials (R)" => "Tutorials_R.md",
    "Tutorials (Python)" => "Tutorials_py.md",
    ]


makedocs(
    sitename="HybridTreeBoosting.jl",
    authors = "Paolo Giordani",
    modules=[HybridTreeBoosting],
    format=Documenter.HTML(
        sidebar_sitename=true, # able or disable the site name on the side bar
        repolink = "https://github.com/PaoloGiordani/HybridTreeBoosting.jl", # explicitly set the remote URL
        edit_link = "main"        # adds an "Edit on GitHub" button on documentation  
      ),
    pages=pages,
    repo = "https://github.com/PaoloGiordani/HybridTreeBoosting.jl.git", # link for edit_link
)

# logo = "assets/logo.png"       # if you have a logo in the assets directory
 
deploydocs(
    repo="PaoloGiordani/HybridTreeBoosting.jl.git", # GitHub complained that I should remove the protocol. Originally repo="https://github.com/PaoloGiordani/HybridTreeBoosting.jl.git"
    target="build",
    branch = "gh-pages",  # 
    devbranch="main",
    push_preview = true,  # false to keep the preview internal 
    )



