#!/usr/bin/env python3
import aws_cdk as cdk
from aws_cdk import BootstraplessSynthesizer

from mlops_infra_stack import MlopsInfraStack


app = cdk.App()

MlopsInfraStack(
    app,
    "MlopsTemplateInfraStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region"),
    ),
    # Avoid CDK bootstrap dependency for restricted org accounts.
    synthesizer=BootstraplessSynthesizer(),
)

app.synth()
