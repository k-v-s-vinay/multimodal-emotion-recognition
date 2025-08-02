# test_mongodb.py
import pymongo
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

def test_mongodb():
    print("Testing MongoDB connection...")
    
    try:
        # Try to connect
        client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ismaster')
        
        # Get server info
        server_info = client.server_info()
        print(f"✅ MongoDB is running!")
        print(f"   Version: {server_info['version']}")
        print(f"   Connection: localhost:27017")
        
        # Test database operations
        db = client.test_database
        collection = db.test_collection
        
        # Insert a test document
        test_doc = {"test": "hello", "status": "working"}
        result = collection.insert_one(test_doc)
        print(f"✅ Database write test passed")
        
        # Read the test document
        found_doc = collection.find_one({"test": "hello"})
        if found_doc:
            print(f"✅ Database read test passed")
        
        # Clean up test document
        collection.delete_one({"test": "hello"})
        print(f"✅ Database delete test passed")
        
        client.close()
        return True
        
    except ServerSelectionTimeoutError:
        print("❌ MongoDB connection timeout - server not responding")
        print("   Make sure MongoDB service is running")
        return False
        
    except ConnectionFailure:
        print("❌ MongoDB connection failed")
        print("   Check if MongoDB is installed and running")
        return False
        
    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
        return False

if __name__ == "__main__":
    test_mongodb()
