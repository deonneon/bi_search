"""
Test script for MicroStrategy Dummy Ingestion

This script tests the basic functionality of the MicroStrategy metadata ingestion
with dummy data to ensure it works correctly.
"""

import unittest
import pandas as pd
from microstrategy_ingestion import MicroStrategyDummyIngestion


class TestMicroStrategyIngestion(unittest.TestCase):
    """Test cases for MicroStrategy ingestion functionality."""
    
    def setUp(self):
        """Set up test client."""
        self.client = MicroStrategyDummyIngestion()
    
    def test_authentication(self):
        """Test authentication functionality."""
        result = self.client.authenticate()
        self.assertTrue(result)
        self.assertTrue(self.client.authenticated)
        self.assertIsNotNone(self.client.session_token)
    
    def test_metadata_generation(self):
        """Test metadata generation for all object types."""
        self.client.authenticate()
        
        # Test individual metadata generation
        reports = self.client.generate_dummy_reports(5)
        self.assertEqual(len(reports), 5)
        self.assertEqual(reports[0]['type'], 'report')
        
        dossiers = self.client.generate_dummy_dossiers(3)
        self.assertEqual(len(dossiers), 3)
        self.assertEqual(dossiers[0]['type'], 'dossier')
        
        datasets = self.client.generate_dummy_datasets(2)
        self.assertEqual(len(datasets), 2)
        self.assertEqual(datasets[0]['type'], 'dataset')
        
        attributes = self.client.generate_dummy_attributes()
        self.assertGreater(len(attributes), 0)
        self.assertEqual(attributes[0]['type'], 'attribute')
        
        metrics = self.client.generate_dummy_metrics(4)
        self.assertEqual(len(metrics), 4)
        self.assertEqual(metrics[0]['type'], 'metric')
    
    def test_full_workflow(self):
        """Test the complete ingestion workflow."""
        # Authenticate
        self.assertTrue(self.client.authenticate())
        
        # Fetch metadata
        metadata = self.client.fetch_all_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertIn('reports', metadata)
        self.assertIn('dossiers', metadata)
        self.assertIn('datasets', metadata)
        self.assertIn('attributes', metadata)
        self.assertIn('metrics', metadata)
        
        # Convert to DataFrame
        df = self.client.metadata_to_dataframe(metadata)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        
        # Check required columns
        required_columns = ['id', 'name', 'description', 'type', 'owner']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Generate summary
        summary = self.client.get_metadata_summary(df)
        self.assertIsInstance(summary, dict)
        self.assertIn('total_objects', summary)
        self.assertIn('object_types', summary)


def run_simple_demo():
    """Run a simple demonstration of the ingestion system."""
    print("=== MicroStrategy Dummy Ingestion Demo ===\n")
    
    # Create client
    client = MicroStrategyDummyIngestion()
    
    # Authenticate
    print("1. Authenticating...")
    if client.authenticate():
        print("   ✓ Authentication successful")
    else:
        print("   ✗ Authentication failed")
        return
    
    # Fetch small sample of metadata
    print("\n2. Generating sample metadata...")
    metadata = {
        "reports": client.generate_dummy_reports(3),
        "dossiers": client.generate_dummy_dossiers(2),
        "datasets": client.generate_dummy_datasets(2),
        "attributes": client.generate_dummy_attributes()[:5],  # First 5 only
        "metrics": client.generate_dummy_metrics(3)
    }
    
    for obj_type, objects in metadata.items():
        print(f"   ✓ Generated {len(objects)} {obj_type}")
    
    # Convert to DataFrame
    print("\n3. Converting to DataFrame...")
    df = client.metadata_to_dataframe(metadata)
    print(f"   ✓ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Show sample data
    print("\n4. Sample Data Preview:")
    print(df[['name', 'type', 'owner', 'status']].head(10))
    
    # Generate summary
    print("\n5. Summary Statistics:")
    summary = client.get_metadata_summary(df)
    print(f"   Total Objects: {summary['total_objects']}")
    print(f"   Object Types: {summary['object_types']}")
    print(f"   Unique Owners: {summary['owners']}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    # Run the simple demo first
    run_simple_demo()
    
    print("\n" + "="*50)
    print("Running Unit Tests...")
    print("="*50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False) 