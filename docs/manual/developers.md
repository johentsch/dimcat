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

1. {func}`~.deserialize_dict`
2. {func}`~.deserialize_config`
3. {func}`~.deserialize_json_str`
4. {func}`~.deserialize_json_file`

This is possible because each deserialized object includes a value for the field `dtype` specifying the object's class name from which the schema can be retrieved thanks to the class attribute {attr}`~.DimcatObject.schema`. Other functions that are relevant in this context are {func}`~.get_class` and {func}`~.get_schema` (see [](#class-registry))

#### Example

```{code-cell}
import dimcat as dc
cfg = dc.DimcatConfig("DimcatObject")
obj = cfg.create()
print("This object is a", type(obj))
json_str = obj.to_json()
obj_copy = dc.base.DimcatObject.from_json(json_str)
another_copy = dc.deserialize_json_str(json_str)
print(f"The two deserialized objects are equivalent: ", obj_copy == another_copy)
print(f"The two deserialized objects are identical: ", obj_copy is another_copy)
obj_copy # DimcatObject.__repr__() uses .to_dict() under the hood
```

#### Implementation

The implementation is centered on two methods of the respective object's nested {class}`~.DimcatSchema` which derives from [marshmallow.Schema](https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html): [schema.dump()](https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html#marshmallow.schema.Schema.dump) and [schema.load()](https://marshmallow.readthedocs.io/en/stable/marshmallow.schema.html#marshmallow.schema.Schema.load). The former takes an object and returns a dictionary, whereas the latter takes a dictionary and returns an object. Correspondingly, {meth}`.DimcatObject.to_dict`  and {meth}`.DimcatObject.from_dict` retrieve the relevant schema singleton from {attr}`.DimcatObject.schema` to call these two methods respectively.

+++

### Creating a new type of DimcatObject

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

In cases where an attribute should point to a DimcatObject (e.g. all {class}`~.Result` objects referencing the analyzed {class}`~.DimcatResource` via the {attr}`~.analyzed_resource` property), we can use the type {class}`~.DimcatObjectField` in the schema.

+++

(class-registry)=
### The class registry

Every {class}`~.DimcatObject` comes with the attribute {attr}`~._registry` which is a dictionary mapping the names of all DimcatObjects to their classes.
It is implemented using [__init_subclass__](https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__).
We don't need to interact directy with the registry thanks to the convenience function {func}`~.get_class` which takes the name of an object as a string and returns the respective class.
In the code, this would typically look like this:

```{code-cell}
Constructor = dc.get_class("FeatureExtractor")
feature_extractor = Constructor()
```

Schemas are not part of the registry. For retrieving a class's schema we can use `Constructor.schema` (building on the example) or the convenience function {func}`~.get_schema`.


### Public and private methods

The DiMCAT project differentiates between private methods whose names begin with `_` and public methods whose names don't.
Semantically, public methods are those that users interact with and which therefore often perform additional checks, e.g. of user input;
then, the public method calls the private method of the same name which performs the actual job.
In most cases, subclasses override only private methods.

#### Example

For example, compare the public {meth}`.PipelineStep.process_dataset` with its private counterpart:

```python
    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        # this is where subclasses create a new package and add it to the dataset
        return new_dataset
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        self.check_dataset(dataset)
        return self._process_dataset(dataset)
```

+++

## Two types of DimcatObjects

All classes that are neither a schema nor a config inherit from one of the two following subclasses of DimcatObject:

1. {class}`~.Data`: a DimcatObject that represents a dataset, a subset of a dataset, or an individual resource such as a dataframe.
2. {class}`~.PipelineStep`: a DimcatObject that accepts a Data object as input and returns a Data object as output.

They are organized in two packages, {mod}`dimcat.data` and {mod}`dimcat.steps`. Objects defined in {mod}`dimcat.steps` operate on objects defined in {mod}`dimcat.data` and can import from it, but not the other way around.
In a few exceptional cases where Data objects need to actively use PipelineStep (which is the case, for example, for {meth}`.Dataset.extract_feature`), we circumvent circular imports by summoning them via {func}`~.get_class` (and not using type hints for the summoned object).

### Data objects

Data is organized into a hierarchical hierarchy of four objects (from top to bottom):

* {class}`~.Dataset`, consisting of two catalogs, called `inputs` and `outputs`;
* {class}`~.DimcatCatalog` ("catalog"), a collection of packages;
* {class}`~.DimcatPackage` ("package"), a collection of resources;
* {class}`~.DimcatResource` ("resource"), a wrapper around a dataframe.

There are three main types of resources, namely

* {class}`~.Facet`, a representation of score-related elements that represents the loaded data with minimal standardization (e.g. {class}`~.MuseScoreNotes`);
* {class}`~.Feature`, some aspect extracted from a facet, standardized by ``Dimcat`` (e.g., {class}`~.Notes`);
* {class}`~.Result`, a result of applying a {class}`~.PipelineStep` to a feature.

``Dimcat`` cannot process facets directly; the relevant features need to be extracted first. Many PipelineSteps extract the required feature automatically.

#### Dataset

The principal Data object is the {class}`~.Dataset` and is the one that users usually interact with the most. Its three principal properties are:

* {attr}`~.inputs`, an {class}`~.InputsCatalog`
* {attr}`~.outputs`, an {class}`~.OutputsCatalog`
* {attr}`~.pipeline`, a {class}`~.Pipeline` consisting of all previously applied {class}`PipelineSteps <~.PipelineStep>`.

After applying a {class}`~.PipelineStep` to a {class}`~.Dataset`,
its {attr}`~.outputs` MUST correspond to the result of applying the {attr}`~.pipeline` to {attr}`~.inputs`.
A serialized Dataset is therefore suited for communicating results in a reproducible manner.
Any {class}`~.PipelineStep` applied on a dataset will be performed on all eligible resources that the packages in {attr}`~.inputs` contain and result in a new dataset containing the relevant output packages/resources under {attr}`~.outputs`.


Datasets are passive 'by nature', meaning that, in general, they are being manipulated by PipelineSteps or by the user.
PipelineSteps process a Dataset by requesting one or several features using {meth}`.Dataset.get_feature`,
 processing each {class}`~.Feature`, and adding the processed Feature(s) to the Dataset's OutputsCatalog. 
However, in one case, the Dataset *does* play an active role, namely in the extraction of features from the InputsCatalog.
When prompted with `.get_feature(F)` where `F` is some specification of a {class}`~.Feature`, the Dataset will

* look up the feature in its OutputCatalog and return it if present,
* call {meth}`.Dataset.extract_feature` otherwise and return its output.

Since the actual extraction happens on the level of a single resource (a {class}`~.Facet` which names the feature among its {attr}`~.DimcatResource.extractable_features`), the latter case invokes the following call chain:

* {meth}`.Dataset.extract_feature` calls
* {meth}`.InputsCatalog.extract_feature` calls
* {meth}`.Package.extract_feature` calls 
* {meth}`.DimcatResource.extract_feature`.

The Dataset applies all previously applied PipelineSteps to the thus extracted {class}`~.Feature`,
 adds it to its {class}`~.OutputsCatalog` and appends the {class}`~.FeatureExtractor` to its {class}`~.Pipeline`.


+++

#### DimcatCatalog

As per the [Frictionless Data specifications](https://frictionlessdata.io), a catalog is a collection of packages. In ``Dimcat``, catalogs appear only in a single place: Every {class}`~.Dataset` consists of two {class}`DimcatCatalogs <.DimcatCatalog>`, namely an {class}`~.InputsCatalog` and an {class}`~.OutputsCatalog`.
{attr}`~.inputs` includes all loaded datapackages, and {attr}`~.outputs` all (processed or unprocessed) features extracted from them, as well as all other 
processing results. 


#### DimcatPackage

The preferred structure of a DimcatPackage is a ``.zip`` and a ``.datapackage.json`` file, where the former contains one or several ``.tsv`` files (resources) described in the latter.
The ``.datapackage.json`` file follows the [Frictionless](https://frictionlessdata.io) Data specifications for packages and allows ``DiMCAT`` to know what's in a package without having to actually load the data from the ``.tsv`` files.

Since the data that DiMCAT transforms and analyzes comes from very heterogeneous sources, each original corpus is pre-processed and stored as a [frictionless.Package](https://framework.frictionlessdata.io/docs/framework/package.html). This task is achieved by loaders. When your aim is to enable ``DiMCAT`` to load a new type of dataset, you will have to implement a new {class}`~.Loader`. Please approach the developers by [creating an issue](https://github.com/DCMLab/dimcat/issues).

#### DimcatResource

From all this follows that the Dataset is mainly a structured container for {class}`DimcatResources <.DimcatResource>`, of which there are three main types:

1. {class}`Facets <.Facet>`, i.e. the resources described in the original datapackage.json. They aim to stay as faithful as possible to the original data, applying only mild standardization and normalization.
   All Facet resources come with several columns that represent timestamps both in absolute and in musical time, allowing for the alignment of different corpora.
2. {class}`Features <.Feature>`, i.e. resources derived from Facets explicitly (via {meth}`.Dataset.extract_feature`) or implicitly (by applying PipelineSteps which extract features automatically). They are standardized objects that are required by  PipelineSteps to compute statistics and visualizations.
   To allow for straightforward serialization of the Dataset, all Feature resources are represented as a DimcatCatalog called {attr}`~.Dataset.outputs`.
3. {class}`Results <.Result>`, i.e., resources that represent the results of applying an {class}`~.Analyzer` to a {class}`~.Feature`. They are equally stored in the {attr}`~.Dataset.outputs` catalog and come with the appropriate methods for plotting the results.

Although [Frictionless resources](https://framework.frictionlessdata.io/docs/framework/resource.html) can be stored as individual ``.tsv`` files with their own ``.resource.json`` descriptors,
 ``DiMCAT`` typically stores multiple resources as a datapackage, i.e., a ``.zip`` file containing one ``.tsv`` file per resource, accompanied with a single ``.datapackage.json`` file detailing the metadata for all of them.
 In both cases, the descriptor specifies the column schema for each resource, allowing ``DiMCAT`` to "lazy-load" the data, meaning that the Dataframes are loaded into memory only the moment when they are actually needed (using {meth}`.DimcatResource.load`).

A DimcatResource can be instantiated in two different ways, either from a descriptor or from a dataframe.
At any given moment, the {attr}`~.DimcatResource.status` attribute returns an Enum value reflecting the availability and state of the/a dataframe. This is relevant for keeping datapackages stored on disk up-to-date. 
When a Dataset is serialized, all dataframes from the outputs catalog that haven't been stored to disk yet are written into one or several .zip files so that they can be referenced by the updated descriptor(s).

If you want to create a new type of DimcatResource, please inherit from the relevant subclass and refer to the docstrings of {class}`~.DimcatResource` in the module {mod}`.dc` for understanding how to use the class variables.

+++



```{code-cell}

```
