## Atlas

The application will start as a basic api that will store and retreive information about a user.


## File Structure
```
├── app                  # "app" is a Python package
│   ├── __init__.py      # this file makes "app" a "Python package"
│   ├── main.py          # "main" module, e.g. import app.main
│   ├── dependencies.py  # "dependencies" module, e.g. import app.dependencies
│   ├── models           # "models" folder containing any required ml models
│   │   ├── __init__.py  # makes "routers" a "Python subpackage"
│   │   └── model.pkl     # "users" submodule, e.g. import app.routers.users
│   └── routers          # "routers" is a "Python subpackage"
│   │   ├── __init__.py  # makes "routers" a "Python subpackage"
│   │   ├── items.py     # "items" submodule, e.g. import app.routers.items
│   │   └── users.py     # "users" submodule, e.g. import app.routers.users
│   └── internal         # "internal" is a "Python subpackage"
│       ├── __init__.py  # makes "internal" a "Python subpackage"
│       └── admin.py     # "admin" submodule, e.g. import app.internal.admin
```

This project will also use Docker to containerize the application and Kubernetes to deploy the application
