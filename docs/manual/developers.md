---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: dimcat
  language: python
  name: dimcat
---

# Developers guide

This guide is an introduction into `DiMCAT's` code architecture. Users who want to contribute to `DiMCAT` are invited to refer to the [contribution guidelines](https://dimcat.readthedocs.io/en/latest/contributing.html) which contain coding conventions and instructions how to set up the development environment.

+++

## Introduction

The library is called DiMCAT and has three high-level objects:

1. {class}`~.DimcatObject` ("object"): the base class for all objects that manages object creation and serialization and subclass registration.
   The DimcatObject class has a class attribute called _registry that is a dictionary of all subclasses of DimcatObject.
   Each DimcatObject has a nested class called {class}`~.DimcatObject.Schema` that inherits from {class}`~.DimcatSchema`.
2. {class}`~.DimcatSchema` ("schema"): the base class for all nested Schema classes, inheriting from [marshmallow.Schema](https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html).
   The Schema defines the valid values ranges for all attributes of the DimcatObject and how to serialize and deserialize them.
3. {class}`~.DimcatConfig` ("config"): a DimcatObject that can represent a subset of the attributes of another DimcatObject and instantiate it using the {meth}`~.DimcatConfig.create` method.
   It derives from MutableMapping and is used for communicating about and checking the compatibility of DimcatObjects.

The three classes are defined in the `src\dimcat\base.py` module.
   

+++

### Serializing `DiMCAT` objects

The nested Schema corresponding to each DimcatObject is instantiated as a singleton and can be retrieved via the class attribute {attr}`~.DimcatObject.schema`.
Using this Schema, a DimcatObject can be serialized to and deserialized from:

1. a dictionary using the {meth}`~.DimcatObject.to_dict` and {meth}`~.DimcatObject.from_dict` methods.
2. a DimcatConfig object using the {meth}`~.DimcatObject.to_config` and {meth}`~.DimcatObject.from_config` methods.
3. a JSON string using the {meth}`~.DimcatObject.to_json` and {meth}`~.DimcatObject.from_json` methods.
4. a JSON file using the {meth}`~.DimcatObject.to_json_file` and {meth}`~.DimcatObject.from_json_file` methods.

Under the hood, methods 2-4 use method 1. In addition, `DiMCAT` has the following standalone functions to deserialize serialized DimcatObjects:

1. {func}`~.base.deserialize_dict`
2. {func}`~.deserialize_config`
3. {meth}`~dimct.base.deserialize_json_str`
4. {meth}`dimct.base.deserialize_json_file`

This is possible because each deserialized object includes a value for the field `dtype` specifying the object's class name from which the schema can be retrieved thanks to the class attribute {attr}`~.DimcatObject.schema`. Other functions that are relevant in this context are {meth}`~.get_class` and {meth}`~.get_schema`.

#### Example

```{code-cell}
import dimcat as dc
cfg = dc.DimcatConfig("DimcatObject")
obj = cfg.create()
print("This object is a", type(obj))
json_str = obj.to_json()
obj_copy = dc.base.DimcatObject.from_json(json_str)
obj_copy # DimcatObject.__repr__() uses .to_dict() under the hood
```

#### Implementation

The implementation is centered on two methods of the respective object's nested {class}`~.DimcatSchema` which derives from [marshmallow.Schema](https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html): [schema.dump()](https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html#marshmallow.schema.Schema.dump) and [schema.load()](https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html#marshmallow.schema.Schema.load). The former takes an object and returns a dictionary, whereas the latter takes a dictionary and returns an object. Correspondingly, {meth}`.DimcatObject.to_dict`  and {meth}`.DimcatObject.from_dict` retrieve the relevant schema singleton from {attr}`.DimcatObject.schema` to call these two methods respectively.
 

+++

#### Creating a new type of DimcatObject

All objects in `DiMCAT` (except {class}`~.DimcatSchema`) inherit from {class}`~.DimcatObject`. Inheritance also concerns the nested schema class. Effectively, this means that if you subclass an existing object type without adding new initialization arguments, your new class can simply inherit its parent's `Schema` class and serialization will just work as described above. However, if you add a property, meaning that you will also need to add the corresponding initialization argument, you also need to include a nested `Schema` class which inherits from the parent's schema. Each property that is to be serialized needs to be declared as [marshmallow field](https://marshmallow.readthedocs.io/en/stable/marshmallow.fields.html) corresponding to the datatype.

```{code-cell}
from marshmallow import fields

class NewType(dc.base.DimcatObject):

    class Schema(dc.base.DimcatObject.Schema):
        new_property = fields.Str()
        
    def __init__(self, new_property: str, **kwargs):
        super().__init__(**kwargs)
        self.new_property = new_property
 
        
new_obj = NewType("some string value")
as_dict = new_obj.to_dict()
new_obj_copy = dc.deserialize_dict(as_dict)
new_obj_copy
```

## Two types of DimcatObjects

All classes that are neither a schema nor a config are one of the two following subclasses of DimcatObject:

1. {class}`~.Data`: a DimcatObject that represents a dataset, a subset of a dataset, or a an individual resource such as a dataframe.
2. {class}`~.PipelineStep`: a DimcatObject that accepts a Data object as input and returns a Data object as output.

**The principal Data object is called {class}`~.Dataset` and is the one that users will interact with the most.**

The Dataset provides convenience methods that are equivalent to applying the corresponding PipelineStep.
Every PipelineStep applied to it will return a new Dataset that can be serialized and deserialized to re-start the pipeline from that point.
To that aim, every Dataset stores a serialization of the applied PipelineSteps and of the original Dataset that served as initial input.
This initial input is specified as a {class}`~.DimcatCatalog` which is a collection of {class}`DimcatPackages <.data.dataset.base.DimcatPackage>`,
each of which is a collection of {class}`DimcatResources <.data.resources.base.DimcatResource>`,
as defined by the [Frictionless Data specifications](https://frictionlessdata.io).
The preferred structure of a DimcatPackage is a .zip and a datapackage.json file, where the former contains one or several .tsv files (resources) described in the latter.
Since the data that DiMCAT transforms and analyzes comes from very heterogeneous sources, each original corpus is pre-processed and stored as a [frictionless.Package](https://framework.frictionlessdata.io/docs/framework/package.html) together with the metadata relevant for reproducing the pre-processing.

+++

**It follows that the Dataset is mainly a container for {class}`DimcatResources <.data.resources.base.DimcatResource>` namely:**

1. Facets, i.e. the resources described in the original datapackage.json. They aim to stay as faithful as possible to the original data, applying only mild standardization and normalization.
   All Facet resources come with several columns that represent timestamps both in absolute and in musical time, allowing for the alignment of different corpora.
   The [Frictionless resource](https://framework.frictionlessdata.io/docs/framework/resource.html) descriptors listed in the datapackage.json contain both the column schema and the piece IDs that are present in each of the facets.
2. {class}`Features <~.data.resources.features.Feature>`, i.e. resources derived from Facets by applying PipelineSteps. They are standardized objects that are requested by the PipelineSteps to  compute statistics and visualizations.
   To allow for straightforward serialization of the Dataset, all Feature resources are represented as a DimcatCatalog called `outputs`, which can be stored as .tsv files in one or several .zip files.

**A {class}`~.DimcatResource` functions similarly to the [frictionless.Resource](https://framework.frictionlessdata.io/docs/framework/resource.html) that it wraps, meaning that it grants access to the metadata without having to load the dataframes into memory.**

It can be instantiated in two different ways, either from a resource descriptor or from a dataframe.
At any given moment, the {attr}`~.DimcatResource.status` attribute returns an Enum value reflecting the availability and state of the/a dataframe.
When a Dataset is serialized, all dataframes from the outputs catalog that haven't been stored to disk yet are written into one or several .zip files so that they can be referenced by resource descriptors.

**One of the most important methods, used by most PipelineSteps, is {meth}`.Dataset.get_feature`, which accepts a Feature config and returns a Feature resource.**

The Feature config is a {class}`~.DimcatConfig` that specifies the type of Feature to be returned and the parameters to be used for its computation. Furthermore, it is also used

1. to determine for each piece in every loaded DimcatPackage an Availability value, ranging from not available over available with heavy computation to available instantly.
2. to determine whether the Feature resource had already been requested and stored in the outputs catalog.

```{code-cell}

```
