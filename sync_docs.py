import os
import shutil
import json
import re

# Configuration of modules with their human-friendly info
MODULES_CONFIG = {
    "/introduction": {
        "src_path": "",
        "dest_name": None,
        "title": "Project Overview",
        "description": "Explore the full Relational Calculus framework and all related research modules.",
        "showStats": False
    },
    "/philosophy": {
        "src_path": "",
        "dest_name": None,
        "title": "Core Philosophy",
        "description": "Deep dive into the theoretical shift from Magnitude to Relation in ML.",
        "showStats": False
    },
    "/core-module": {
        "src_path": "relational_calculus",
        "dest_name": "relational_calculus.md",
        "title": "Core Architecture",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/research/emergent": {
        "src_path": "emergent_intelligence",
        "dest_name": "emergent_intelligence.md",
        "title": "Emergent Research",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/research/green-ai": {
        "src_path": "green_AI_practitioner_guide",
        "dest_name": "green_AI_practitioner_guide.md",
        "title": "Green AI Practitioner",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/research/quantum": {
        "src_path": "quantum_machine_learning",
        "dest_name": "quantum_machine_learning.md",
        "title": "Quantum ML Lab",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/research/lidar": {
        "src_path": "relational_lidar",
        "dest_name": "relational_lidar.md",
        "title": "Relational Lidar",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/research/rna": {
        "src_path": "rna_sequencing",
        "dest_name": "rna_sequencing.md",
        "title": "Genomics Lab",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/examples/core": {
        "src_path": "use_examples/1_core_architecture",
        "dest_name": "use_examples_1_core_architecture.md",
        "title": "Core Optimization",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/examples/physics": {
        "src_path": "use_examples/2_physics_and_continuous_systems",
        "dest_name": "use_examples_2_physics_and_continuous_systems.md",
        "title": "Physics & Fluid",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/examples/robotics": {
        "src_path": "use_examples/3_robotics_and_vision",
        "dest_name": "use_examples_3_robotics_and_vision.md",
        "title": "Robotics Lab",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/examples/hep": {
        "src_path": "use_examples/hep",
        "dest_name": "use_examples_hep.md",
        "title": "High Energy Physics",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/examples/nlp": {
        "src_path": "use_examples/nlp_and_enterprise_ai",
        "dest_name": "use_examples_nlp_and_enterprise_ai.md",
        "title": "Enterprise NLP",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    },
    "/examples/tabular": {
        "src_path": "use_examples/tabular_data_xgboost",
        "dest_name": "use_examples_tabular_data_xgboost.md",
        "title": "Tabular XGBoost",
        "description": "Access the technical implementations, datasets, and research papers for this module.",
        "showStats": True
    }
}

def count_files(directory):
    stats = {"scripts": 0, "papers": 0, "others": 0}
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return stats
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == "readme.md" or file.lower() == "__init__.py" or file.startswith("."):
                continue
            ext = os.path.splitext(file)[1].lower()
            name = file.lower()
            if "paper" in name:
                stats["papers"] += 1
            elif ext == ".py":
                stats["scripts"] += 1
            else:
                stats["others"] += 1
    return stats

def fix_image_paths(content, route):
    # Calculate depth of the route (e.g., /research/green-ai has depth 2)
    # Depth 1 -> assets/
    # Depth 2 -> ../assets/
    # Depth 3 -> ../../assets/
    depth = route.count('/') - 1
    prefix = "../" * depth if depth > 0 else ""
    
    # Replaces ![alt](../docs/assets/name.png) or similar with ![alt](prefix + assets/name.png)
    return re.sub(r'\((\.\.\/)+docs\/assets\/', f'({prefix}assets/', content)

def sync():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "docs", "modules")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    dynamic_metadata = {}

    for route, info in MODULES_CONFIG.items():
        src_path = os.path.join(base_dir, info["src_path"])
        
        if info["dest_name"]:
            readme_src = os.path.join(src_path, "README.md")
            readme_dest = os.path.join(target_dir, info["dest_name"])
            if os.path.exists(readme_src):
                # Read, Fix Paths based on Route Depth, and Write
                with open(readme_src, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                fixed_content = fix_image_paths(content, route)
                
                with open(readme_dest, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                print(f"[✓] Synced & Depth-Fixed README for {route}")
        
        stats = count_files(src_path) if info["src_path"] else {"scripts": 0, "papers": 0, "others": 0}
        dynamic_metadata[route] = {
            "path": info["src_path"],
            "title": info["title"],
            "description": info["description"],
            "showStats": info["showStats"],
            "scripts": stats["scripts"],
            "papers": stats["papers"],
            "others": stats["others"]
        }

    js_content = f"window.labMetadata = {json.dumps(dynamic_metadata, indent=2)};"
    with open(os.path.join(target_dir, "metadata.js"), "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"[✓] Generated dynamic metadata.js")

if __name__ == "__main__":
    sync()
