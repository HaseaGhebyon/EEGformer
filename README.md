<div align="center">
<a href="https://refine.dev/">
    <img alt="refine logo" src="https://refine.ams3.cdn.digitaloceanspaces.com/readme/refine-readme-banner.png">
</a>

<br/>
<br/>

<div align="center">
    <a href="https://refine.dev">Home Page</a> |
    <a href="https://refine.dev/docs/">Documentation</a> |
    <a href="https://refine.dev/examples/">Examples</a> |
    <a href="https://discord.gg/refine">Discord</a> |
    <a href="https://refine.dev/blog/">Blog</a>
</div>
</div>

<br/>
<br/>

<div align="center"><strong>The sweet spot between the low/no code and “starting from scratch” for CRUD-heavy applications.</strong><br> Refine is as an open source, React meta-framework for enterprise. It provides a headless solution for everything from admin panels to dashboards and internal tools.
<br />
<br />

</div>

<div align="center">

[![Awesome](https://github.com/refinedev/awesome-refine/raw/main/images/badge.svg)](https://github.com/refinedev/awesome-refine)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8101/badge)](https://www.bestpractices.dev/projects/8101)
[![npm version](https://img.shields.io/npm/v/@refinedev/core.svg)](https://www.npmjs.com/package/@refinedev/core)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

[![Discord](https://img.shields.io/discord/837692625737613362.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/refine)
[![Twitter Follow](https://img.shields.io/twitter/follow/refine_dev?style=social)](https://twitter.com/refine_dev)

</div>

<br/>

[![how-refine-works](https://refine.ams3.cdn.digitaloceanspaces.com/website/static/img/diagram-3.png)](https://refine.dev)

## What is Refine?

**Refine** is a React meta-framework for CRUD-heavy web applications. It addresses a wide range of enterprise use cases including internal tools, admin panels, dashboards and B2B apps.

Refine's core hooks and components streamline the development process by offering industry-standard solutions for crucial aspects of a project, including **authentication**, **access control**, **routing**, **networking**, **state management**, and **i18n**.

Refine's headless architecture enables the building of highly customizable applications by decoupling business logic from UI and routing. This allows integration with:

- Any custom designs or UI frameworks like [TailwindCSS](https://tailwindcss.com/), along with built-in support for [Ant Design](https://ant.design/), [Material UI](https://mui.com/material-ui/getting-started/overview/), [Mantine](https://mantine.dev/), and [Chakra UI](https://chakra-ui.com/).

- Various platforms, including Next.js, Remix, React Native, Electron, etc., by a simple routing interface without the need for additional setup steps.

## ⚡ Try Refine

Start a new project with Refine in seconds using the following command:

```sh
npm create refine-app@latest my-refine-app
```

Or you can create a new project on your browser:

<a href="https://refine.dev/?playground=true" target="_blank">
  <img height="48" width="245" src="https://refine.ams3.cdn.digitaloceanspaces.com/assets/try-it-in-your-browser.png" />
</a>

## Konfigurasi

Here is the configuration file (found at <a href="./util/config.py">./util/config.py</a>). It is used for the entire code, so please check it carefully when you want to run the training.

```python
def get_config():
    return {
        # MODEL & TRAINER CONFIGURATION
        "batch_size" : 32,
        "num_epoch" : 500,
        "learning_rate" : 1e-04,
        "epsilon" : 1e-9,
        "num_cls" : 5,
        "transformer_size" : 1,

        "root" : "/raid/data/m13520079/EEGformer",
        "datasource" : "21chan_5st_120dp_20step",
        "model_folder" : "weights",
        "model_basename" : "eegformer_model",
        "experiment_name": "runs/eegformermodel",
        "preload" : "latest",

        "channel_order" : ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],        
        "selected_channel" : ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],
        "seq_len" :120,
        "sliding_step": 20,

        # DATASET CONFIGURATION
        "dataset_dir" : "/raid/data/m13520079/EEGformer/dataset",
        "low_cut" : 0.57, # Hz
        "high_cut" : 70.0, # Hz
        "samp_freq" : 200.0,
        "bandpass_order" : 1,
        "datasample_per_label" : 200,
        "img_dataset_dir" : "imgdataset"
    }
```

The result will look like this:

[![Refine + Material UI Example](https://refine.ams3.cdn.digitaloceanspaces.com/assets/refine-mui-simple-example-screenshot-rounded.webp)](https://refine.new/preview/c85442a8-8df1-4101-a09a-47d3ca641798)

## Use cases

**Refine** shines on _data-intensive⚡_ enterprise B2B applications like **admin panels**, **dashboards** and **internal tools**. Thanks to the built-in **SSR support**, it can also power _customer-facing_ applications like **storefronts**.

You can take a look at some live examples that can be built using **Refine** from scratch:

- [Fully-functional CRM Application](https://refine.dev/templates/crm-application/)
- [Fully-functional Admin Panel](https://refine.dev/templates/react-admin-panel/)
- [Win95 Style Admin panel 🪟](https://refine.dev/templates/win-95-style-admin-panel/)
- [PDF Invoice Generator](https://refine.dev/templates/react-pdf-invoice-generator/)
- [Medium Clone - Real World Example](https://refine.dev/templates/react-crud-app/)
- [Multitenancy Example](https://refine.dev/templates/multitenancy-strapi/)
- [Storefront](https://refine.dev/templates/next-js-ecommerce-store/)
- [Refer to templates page for more examples](https://refine.dev/templates/)
- [More **Refine** powered different usage scenarios can be found here](https://refine.dev/docs/examples#other-examples)

## Key Features

- Refine Devtools - dive deeper into your app and provide useful insights
- Connectors for **15+ backend services** including [REST API](https://github.com/refinedev/refine/tree/master/packages/simple-rest), [GraphQL](https://github.com/refinedev/refine/tree/master/packages/graphql), [NestJs CRUD](https://github.com/refinedev/refine/tree/master/packages/nestjsx-crud), [Airtable](https://github.com/refinedev/refine/tree/master/packages/airtable), [Strapi](https://github.com/refinedev/refine/tree/master/packages/strapi), [Strapi v4](https://github.com/refinedev/refine/tree/master/packages/strapi-v4), [Supabase](https://github.com/refinedev/refine/tree/master/packages/supabase), [Hasura](https://github.com/refinedev/refine/tree/master/packages/hasura), [Appwrite](https://github.com/refinedev/refine/tree/master/packages/appwrite), [Nestjs-Query](https://github.com/refinedev/refine/tree/master/packages/nestjs-query), [Firebase](https://firebase.google.com/), [Sanity](https://www.sanity.io/), and [Directus](https://directus.io/).
- SSR support with Next.js & Remix and Advanced routing with any router library of your choice
- Auto-generation of CRUD UIs based on your API data structure
- Perfect state management & mutations with React Query
- Providers for seamless authentication and access control flows
- Out-of-the-box support for live / real-time applications
- Easy audit logs & document versioning

## Learning Refine

- Navigate to the [Tutorial](https://refine.dev/docs/tutorial/introduction/index/) on building comprehensive CRUD application guides you through each step of the process.
- Visit the [Guides & Concepts](https://refine.dev/docs/guides-concepts/general-concepts/) to get informed about the fundamental concepts.
- Read more on [Advanced Tutorials](https://refine.dev/docs/advanced-tutorials/) for different usage scenarios.

## Contribution

[Refer to the contribution docs for more information.](https://refine.dev/docs/contributing/#ways-to-contribute)

If you have any doubts related to the project or want to discuss something, then join our [Discord server](https://discord.gg/refine).

## Contributors ♥️ Thanks

We extend our gratitude to all our numerous contributors who create plugins, assist with issues and pull requests, and respond to questions on Discord and GitHub Discussions.

Refine is a community-driven project, and your contributions continually improve it.

<br/>

<a href="https://github.com/refinedev/refine/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=refinedev/refine&max=400&columns=20" />
</a>

## License

Licensed under the MIT License, Copyright © 2021-present Refinedev