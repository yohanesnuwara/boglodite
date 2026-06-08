# Boglodite

![logo](./assets/logo.png)

Boglodite is an agent for subsurface and geoscience. There are 100+ open source repositories around geoscience and Boglodite will make it easy for geoscientists to work with these open source tools, thanks to agent 🤖 

See existing tools in the gallery below. You can also add your tool to Boglodite. 

![malenov-og-faultseg](/assets/F3_fault_vs_facies_inline130.png)

## Work with your favorite CLI

Instruction coming soon

### Copilot CLI

```mermaid
flowchart TD
    User([User])

    subgraph CLI["Copilot CLI"]
        Copilot["GitHub Copilot Agent"]
        Instructions[".github/copilot-instructions.md - Agent System Prompt (orchestrates everything)"]
    end

    Instructions -. governs .-> Copilot
    User -->|"interacts via chat"| Copilot
    Copilot -->|"/skill"| Loader{{Skill Loader}}

    subgraph Skills["Prebuilt Skills Registry"]
        direction LR
        Init["initiate_boglodite"]
        AddTool["add_geo_tool"]
        Malenov["malenov"]
        FaciesNet["faciest_net"]
        FaultSeg["faultseg"]
    end

    Loader -->|"loads"| Skills

    Init -->|"/initiate_boglodite"| Setup
    subgraph Setup["Repository Setup"]
        direction TB
        Dirs["Create directory structure"]
        F3[("Download F3 seismic data from Google Drive")]
    end

    AddTool -->|"/add_geo_tool"| Libs["Register user's favorite Python geoscience libraries"]

    Malenov -->|"/malenov"| MalOut["Seismic facies segmentation"]
    FaciesNet -->|"/faciest_net"| FNOut["Facies classification (network model)"]
    FaultSeg -->|"/faultseg"| FaultOut["Fault extraction / segmentation"]

    classDef core fill:#1e3a5f,stroke:#4a90d9,color:#fff,stroke-width:2px;
    classDef skill fill:#2d4a2b,stroke:#6ab04c,color:#fff;
    classDef action fill:#3d2d52,stroke:#9b59b6,color:#fff;
    classDef data fill:#d9d9d9,stroke:#999999,color:#000;

    class Copilot,Instructions,Loader core;
    class Init,AddTool,Malenov,FaciesNet,FaultSeg skill;
    class Dirs,Libs,MalOut,FNOut,FaultOut action;
    class F3 data;
```

### Claude Code

Support coming soon

### Opencode

Support coming soon

## Gallery

By default, Boglodite supports the following tools and workflows, each with its own `SKILL.md`.

| Name | Skill | Description |
|---|---|---|
| [MalenoV](https://github.com/bolgebrygg/MalenoV) | [SKILL.md](./skills/MalenoV/SKILL.md) | 3D CNN-based seismic facies classification on SEGY volumes using voxel inputs. |
| [facies_net](https://github.com/crild/facies_net) | [SKILL.md](./skills/facies_net/SKILL.md) | Companion to MalenoV, Modular seismic facies classification with data augmentation, TensorBoard logging, and pre-trained models. |
| [faultSeg](https://github.com/xinwucwp/faultSeg) | [SKILL.md](./skills/faultSeg/SKILL.md) | 3D U-Net for automatic seismic fault segmentation, trained on synthetic data and applied to real field volumes (Wu et al., 2019). |

<details>
  <summary><b>MalenoV</b></summary>

  <img width="700" height="200" alt="Image" src="./assets/malenov.png" />

</details>