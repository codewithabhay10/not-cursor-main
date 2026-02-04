Here is the full file content:

```Python
from flask import request, jsonify
from app.models.todo import TodoModel

class TodoController:
    def get_all_todos(self):
        todos = TodoModel.get_all()
        return jsonify([todo.to_dict() for todo in todos])

    def get_todo(self, id):
        todo = TodoModel.get(id)
        if todo is None:
            return jsonify({'error': 'Todo not found'}), 404
        return jsonify(todo.to_dict())

    def create_todo(self):
        data = request.get_json()
        todo = TodoModel(**data)
        todo.save()
        return jsonify(todo.to_dict()), 201

    def update_todo(self, id):
        data = request.get_json()
        todo = TodoModel.get(id)
        if todo is None:
            return jsonify({'error': 'Todo not found'}), 404
        for key, value in data.items():
            setattr(todo, key, value)
        todo.save()
        return jsonify(todo.to_dict())

    def delete_todo(self, id):
        TodoModel.delete(id)
        return jsonify({'message': 'Todo deleted successfully'})
```

In this controller class, there are four methods: 

1. `get_all_todos`: Returns a list of all todos in the database.
2. `get_todo`: Retrieves a specific todo by its ID.
3. `create_todo`: Creates a new todo based on the data provided in the request body.
4. `update_todo`: Updates an existing todo with the new data provided in the request body.
5. `delete_todo`: Deletes a specific todo by its ID.

Each method handles the corresponding HTTP requests and returns a JSON response.