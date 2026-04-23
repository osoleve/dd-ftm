"""Plugin definition for the NeMo Data Designer name-to-IPA processor."""

from data_designer.plugins import Plugin, PluginType

plugin = Plugin(
    config_qualified_name="dd_name_ipa.processor.config.NameIpaProcessorConfig",
    impl_qualified_name="dd_name_ipa.processor.impl.NameIpaProcessor",
    plugin_type=PluginType.PROCESSOR,
)
